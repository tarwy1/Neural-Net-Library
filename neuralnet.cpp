#define _WIN32_WINNT 0X0501
#include <vector>
#include <iostream>
#include "cmath"
#include <algorithm>
#include <fstream>
#include <string>
#include <random>
#include <ctime>
#include "./mnist-master/mnist-master/include/mnist/mnist_reader.hpp"
#include "mingw.thread.h"

using namespace std;

class NN{
    public:
    
        float Sigmoid(float x){ return 1/(1+exp(-x)); }
        float DSigmoid(float x){ return Sigmoid(x) * (1-Sigmoid(x)); }
        float Logit(float x){ return log(x/(1-x)); }
        float Tanh(float x){ return tanhf(x); }
        float ATanh(float x){ return 0.5f * log((1 + x)/(1 - x)); }
        float DTanh(float x){
            float one = tanhf(x);
            return 1 - one*one;
        }
        float LeakyRelu(float x){
            if(x > 0){ return x; }
            return 0.1*x;
        }
        float DLeakyRelu(float x){
            if(x > 0){ return 1; }
            return 0.1;
        }
        float Relu(float x){
            if(x > 0){ return x; }
            return 0;
        }
        float DRelu(float x){
            if(x > 0){ return 1; }
            return 0;
        }
        float Softmax(float x, int id){
            float exps = 0;
            for(int i = layerStarts[Nodes[id].layer]; i < Nodes.size(); i++){
                exps += exp(Nodes[i].z-maxoutput);
            }
            return exp(x-maxoutput)/exps;
        }
        float Activation(float x, int id){
            if(LayerActivationNums[Nodes[id].layer] == 0){ return Sigmoid(x); }
            if(LayerActivationNums[Nodes[id].layer]==1){ return Relu(x); }
            if(LayerActivationNums[Nodes[id].layer]==2){ return LeakyRelu(x); }
            if(LayerActivationNums[Nodes[id].layer]==3){ return Tanh(x); }
            if(LayerActivationNums[Nodes[id].layer]==4){ return Softmax(x, id); }
            return 0;
        }
        float DActivation(float x, int id){
            if(LayerActivationNums[Nodes[id].layer]==0){ return DSigmoid(x); }
            if(LayerActivationNums[Nodes[id].layer]==1){ return DRelu(x); }
            if(LayerActivationNums[Nodes[id].layer]==2){ return DLeakyRelu(x); }
            if(LayerActivationNums[Nodes[id].layer]==3){ return DTanh(x); }
            return 0;
        }
        float InverseActivation(float x){
            if(FunctionNum==0){ return Logit(x); }
            if(FunctionNum==1){
                if(x>0){return x;}
                return 0;
            }
            if(FunctionNum==2){
                if(x>0){return x;}
                return x/0.1;
            }
            if(FunctionNum==3){ return ATanh(x); }
            return 0;
        }
        
        float mse(float x, float intended){
            return ((intended - x) * (intended - x));
        }
        float Dmse(float x, float intended){
            return -2 * (intended - x);
        }
        float binarycrossentropy(float x, float intended){
            return ((intended * log(x)) + ((1-intended) * log(1 - x)));
        }
        float Dbinarycrossentropy(float x, float intended){
            return -((intended/x) + (intended-1)/(1-x));
        }
        float CostFunction(float x, float intended){
            if(CostFunctionNum==0){ return mse(x, intended); }
            else if(CostFunctionNum==1){ return binarycrossentropy(x, intended); }
            return 0;
        }
        float DCostFunction(float x, float intended){
            if(CostFunctionNum==0){ return Dmse(x, intended); }
            else if(CostFunctionNum==1){ return Dbinarycrossentropy(x, intended); }
            return 0;
        }

        struct Node{
            int id; // node id (just an integer)
            int layer; // layer its in,  0 is the input layer
            int numInNodes; // number of nodes in the previous layer
            bool isInput; // if the node is in the input layer
            vector<int> inNodes; // list of the id's of nodes in previous layer
            vector<float> inWeights; // list of values of the weights between nodes in previous layer
            float z; // z
            float a; // a  (the one after sigmoid)
            float bias;
        };

        int TL;
        vector<int> NNL;
        vector<Node> Nodes;
        vector<vector<float>> inVal;
        vector<vector<float>> outVal;
        int datapiece;
        string Function;
        int FunctionNum = 0;
        string CostFunctionStr;
        int CostFunctionNum = 0;
        float BiasMult = 1.0f;
        string OptimizerStr;
        int OptimizerNum = 0;
        vector<int> layerStarts;
        float momentumConst = 0.9f;
        vector<vector<float>> updateVectorWeights;
        vector<float> updateVectorBiases;
        float adjustRate = 0.0001f;
        bool updateVectorsInitialized = false;
        vector<string> LayerActivationStrings;
        vector<int> LayerActivationNums;
        float maxoutput = 0.0f;
        int threadNum;

        NN(vector<int> _NNL, string _Function, string _CostFunctionStr, string _OptimizerStr){
            threadNum = 100;
            srand(time(0));
            Function = _Function;
            CostFunctionStr = _CostFunctionStr;
            OptimizerStr = _OptimizerStr;
            TL = _NNL.size();
            NNL = _NNL;
            int idcount = 0;
            if(Function=="Sigmoid"){ FunctionNum = 0; }
            else if(Function=="Relu"){ FunctionNum = 1; }
            else if(Function=="LeakyRelu"){ FunctionNum = 2; }
            else if(Function=="Tanh"){ FunctionNum = 3; }
            else{ std::cout << "Invalid activation function inputted"; return; }
            if(CostFunctionStr=="mse"){ CostFunctionNum = 0;}
            else if(CostFunctionStr=="binary crossentropy"){ CostFunctionNum = 1;}
            else{ std::cout << "Invalid cost function inputted"; return; }
            if(OptimizerStr=="sgd"){ OptimizerNum = 0; }
            else if(OptimizerStr=="Msgd"){ OptimizerNum = 1; }
            else{ std::cout << "Invalid optimizer inputted"; return; }
            for(int i = 0; i < TL; i++){
                LayerActivationStrings.push_back(Function);
                LayerActivationNums.push_back(FunctionNum);
            }
            std::default_random_engine generator;
            for(int i = 0; i < NNL.size(); i++){
                for(int k = 0; k < NNL[i]; k++){
                    Node nod;
                    nod.id = idcount;
                    nod.layer = i;
                    nod.isInput = !(bool)i;
                    nod.numInNodes = 0;
                    if(!nod.isInput){
                        nod.numInNodes = NNL[i-1];
                        std::normal_distribution<float> gaussian(0.0f, sqrt(2.0f/nod.numInNodes));
                        std::uniform_real_distribution<float> uniform(-1.0f/sqrt(nod.numInNodes), 1.0f/sqrt(nod.numInNodes));
                        for(int j = 0; j < Nodes.size(); j++){
                            if(Nodes[j].layer == i-1){
                                nod.inNodes.push_back(j);
                                if(FunctionNum == 1 || FunctionNum == 2 || FunctionNum == 3){ nod.inWeights.push_back(gaussian(generator)); }
                                if(FunctionNum == 0){ nod.inWeights.push_back(uniform(generator)); }
                            }
                        }
                    }
                    nod.a = 0;
                    nod.z = 0;
                    nod.bias = 0;
                    Nodes.push_back(nod);
                    idcount++;
                }
            }
            int prevLayer = -1;
            for(int i = 0; i < Nodes.size(); i++){
                if(Nodes[i].layer!=prevLayer){
                    layerStarts.push_back(i);
                    prevLayer = Nodes[i].layer;
                }
            }
            layerStarts.push_back(Nodes.size());
        }

        void UpdateLayerActivation(NN& net, string activation, int layer){
            if(activation=="Sigmoid"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 0; }
            else if(activation=="Relu"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 1; }
            else if(activation=="LeakyRelu"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 2; }
            else if(activation=="Tanh"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 3; }
            else if(activation=="Softmax" && layer==net.TL-1){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 4; }
            else{ std::cout << "Activation function inputted incorrectly" << "\n";}
        }

        void DataShuffle(NN& net){
            vector<vector<float>> in = net.inVal;
            vector<vector<float>> out = net.outVal;
            vector<int> ids;
            for(int i = 0; i < in.size(); i++){ ids.push_back(i); }
            std::random_shuffle(ids.begin(), ids.end());
            for(int i = 0; i < ids.size(); i++){
                net.inVal[i] = in[ids[i]];
                net.outVal[i] = out[ids[i]];
            }
        }

        void parrallelActivation(NN& net, NN::Node& nod){
            if(nod.isInput){ return; }
            nod.z = net.BiasMult * nod.bias;
            for(int i = 0; i < nod.numInNodes; i++){
                nod.z += nod.inWeights[i] * net.Nodes[i + nod.inNodes[0]].a;
            }
            //nod.a = net.Activation(nod.z, nod.id);
        }

        void ForwardProp(NN& net){
            int prevlayer = 0;
            for(int NODE = 0; NODE < net.Nodes.size(); NODE++){
                if(net.Nodes[NODE].layer!=prevlayer){
                    prevlayer = net.Nodes[NODE].layer;
                    for(int i = net.layerStarts[prevlayer-1]; i < net.layerStarts[prevlayer]; i++){
                        net.Nodes[i].a = net.Activation(net.Nodes[i].z, i);
                    }
                }
                net.parrallelActivation(net, net.Nodes[NODE]);
            } 
            for(int i = net.layerStarts[net.TL-1]; i < net.Nodes.size(); i++){
                net.Nodes[i].a = net.Activation(net.Nodes[i].z, i);
            }
        }

        void loadValue(int _datapiece, NN& net){
            net.datapiece = _datapiece;
            for(int i = 0; i < net.NNL[0]; i++){
                if(net.Nodes[i].isInput = true){
                    net.Nodes[i].z = net.inVal[datapiece][i] + net.Nodes[i].bias*net.BiasMult;
                    net.Nodes[i].a = Activation(net.Nodes[i].z, i);
                }
            }
        }

        vector<float> predict(NN& net, vector<float> Input){
            for(int i = 0; i < net.NNL[0]; i++){
                net.Nodes[i].z = Input[i]+ net.Nodes[i].bias*net.BiasMult;
                net.Nodes[i].a = net.Activation(net.Nodes[i].z, i);
            }
            net.ForwardProp(net);
            vector<float> returnVals;
            for(int i = 0; i < net.Nodes.size(); i++){
                if(net.Nodes[i].layer == net.TL-1){
                    returnVals.push_back(net.Nodes[i].a);
                }
            }
            return returnVals;
        }

        void printPrediction(NN& net, vector<float> Input){
            vector<float> outs = predict(net, Input);
            for(int i = 0; i < outs.size(); i++){
                std::cout << Input[i] << " " << outs[i] << "\n";
            }
        }

        void UpdateParams(NN& net, vector<vector<float>> &WeightChanges, vector<float> &BiasChanges){
            if(net.OptimizerNum==0){
                for(int i = 0; i < net.NNL[0]; i++){
                    net.Nodes[i].bias-=net.adjustRate*BiasChanges[i];
                    BiasChanges[i] = 0;
                }
                for(int NODE = net.NNL[0]; NODE < net.Nodes.size(); NODE++){
                    net.Nodes[NODE].bias -= net.adjustRate * BiasChanges[NODE];
                    BiasChanges[NODE] = 0;
                    for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                        net.Nodes[NODE].inWeights[i] -= net.adjustRate * WeightChanges[NODE][i];
                        WeightChanges[NODE][i] = 0;
                    }
                }                
            }
            if(net.OptimizerNum==1){
                float update;
                for(int i = 0; i < net.NNL[0]; i++){
                    update = net.adjustRate*BiasChanges[i] + net.momentumConst*net.updateVectorBiases[i];
                    net.Nodes[i].bias-= update;
                    net.updateVectorBiases[i] = update;
                    BiasChanges[i] = 0;
                }
                for(int NODE = net.NNL[0]; NODE < net.Nodes.size(); NODE++){
                    update = net.adjustRate*BiasChanges[NODE] + net.momentumConst*net.updateVectorBiases[NODE];
                    net.Nodes[NODE].bias -= update;
                    net.updateVectorBiases[NODE] = update;
                    BiasChanges[NODE] = 0;
                    for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                        update = net.adjustRate * WeightChanges[NODE][i] + net.momentumConst*net.updateVectorWeights[NODE][i];
                        net.Nodes[NODE].inWeights[i] -= update;
                        net.updateVectorWeights[NODE][i] = update;
                        WeightChanges[NODE][i] = 0;
                    }
                }  
            }
        }

        void Train(NN& net, vector<vector<float>> InData, vector<vector<float>> OutData, int batch, int epochs, float LR){
            net.inVal = InData;
            net.outVal = OutData;
            net.DataShuffle(net);
            float cost = 0;
            float avgcost = 0;
            net.adjustRate = LR/batch;
            int firstOutput = net.Nodes[net.Nodes.size() - 1].inNodes[net.Nodes[net.Nodes.size()-1].inNodes.size() - 1] + 1;
            vector<vector<float>> changesWeights;
            vector<float> changesBiases;
            vector<float> baseDerivatives;
            for(int i = 0; i < net.Nodes.size(); i++){
               baseDerivatives.push_back(0.0f);
            }
            for(int epoch = 0; epoch < epochs; epoch++){
                net.DataShuffle(net);
                for(int point = 0; point < net.inVal.size(); point++){
                    net.loadValue(point, net);
                    net.ForwardProp(net);
                    for(int i = 0; i < net.outVal[point].size(); i++){
                        cost += CostFunction(net.Nodes[firstOutput + i].a, net.outVal[point][i]);
                    }
                    avgcost+=cost;
                    cost = 0;
                    if(epoch == 0 && point == 0){
                        for(int NODE = 0; NODE < net.Nodes.size(); NODE++){
                            changesBiases.push_back(0.0f);
                            if(!net.updateVectorsInitialized){net.updateVectorBiases.push_back(0.0f);}
                            changesWeights.push_back({});
                            if(!net.updateVectorsInitialized){net.updateVectorWeights.push_back({});}
                            for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                                changesWeights[NODE].push_back(0.0f);
                                if(!net.updateVectorsInitialized){net.updateVectorWeights[NODE].push_back(0.0f);}
                            }
                        } 
                        net.updateVectorsInitialized = true;
                    }

                    for(int NODE = net.Nodes.size()-1; NODE >= 0; NODE--){
                        if(net.Nodes[NODE].layer==net.TL-1){
                            baseDerivatives[NODE] = DCostFunction(net.Nodes[NODE].a, net.outVal[point][NODE-firstOutput]) * DActivation(net.Nodes[NODE].z, NODE);
                        }
                        else{
                            baseDerivatives[NODE] = 0;
                            for(int i = 0; i < net.NNL[net.Nodes[NODE].layer + 1]; i++){
                                baseDerivatives[NODE]+=(baseDerivatives[net.layerStarts[net.Nodes[NODE].layer + 1] + i] * net.Nodes[net.layerStarts[net.Nodes[NODE].layer + 1] + i].inWeights[NODE - net.Nodes[net.layerStarts[net.Nodes[NODE].layer + 1] + i].inNodes[0]]);
                            }
                            baseDerivatives[NODE] *= DActivation(net.Nodes[NODE].z, NODE);
                        }
                        changesBiases[NODE] += baseDerivatives[NODE] * BiasMult;
                        if(net.Nodes[NODE].layer>0){
                            for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                                changesWeights[NODE][i] += baseDerivatives[NODE] * net.Nodes[net.Nodes[NODE].inNodes[i]].a;
                            }
                        }
                    }

                    if(point%batch==0 && point > 0){
                        net.UpdateParams(net, changesWeights, changesBiases);
                    }
                }
                if(net.inVal.size()%batch!=0){
                    net.UpdateParams(net, changesWeights, changesBiases);
                }
                std::cout << "cost: " << avgcost/net.inVal.size() << " Epoch: " << epoch << "\n";
                avgcost = 0;
            }   
        }
};


int main(){

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./mnist-master/mnist-master/");

    NN Network({784, 64, 32, 10}, "LeakyRelu", "mse", "Msgd");
    //Network.UpdateLayerActivation(Network, "Softmax", 3);

    vector<vector<float>> inData;
    vector<vector<float>> outData;
    for(int i = 0; i < 1000; i++){
        inData.push_back({});
        for(int j = 0; j < 784; j++){ 
            inData[i].push_back(((float)dataset.training_images[i][j])/255);
        }
        outData.push_back({});
        for(int j = 0; j < 10; j++){
            if(j==(int)dataset.training_labels[i]){
                outData[i].push_back(1.0f);
            }
            else{
                outData[i].push_back(0.0f);
            }
        }
    }
    Network.inVal = inData;
    Network.outVal = outData;

    // for(float i = -10; i < 10; i += 0.001){
    //     Network.inVal.push_back({i});
    //     Network.outVal.push_back({Network.Activation(1/abs(i)*sinf(i)*sinf(i))});
    // }

    Network.Train(Network, Network.inVal, Network.outVal, 32, 20, 0.01f);

    vector<vector<float>> testingData;
    vector<float> testingAnswers;
    for(int i = 0; i < 10000; i++){
        testingData.push_back({});
        for(int j = 0; j < 784; j++){
            testingData[i].push_back((float)dataset.test_images[i][j]/255);
        }
        testingAnswers.push_back((int)dataset.test_labels[i]);
    }
    float totalRight = 0;
    for(int i = 0; i < 10000; i++){
        vector<float> prediction = Network.predict(Network, testingData[i]);
        //for(int i = 0; i < 10; i++){
        //    std::cout << prediction[i] << " "; 
        //}
        //std::cout << "\n";
        int maxindex = 0;
        for(int i = 0; i < 10; i++){
            if(prediction[i] >prediction[maxindex]){
                maxindex = i;
            }
        }
        if((int)maxindex==(int)testingAnswers[i]){
            totalRight++;
        }
    }
    std::cout << totalRight/10000*100 << "\n";
    
    return 0;
}
