#include "neuralnet.hpp"

using namespace std;

// activation functions and derivatives and inverses(for some) D=derivative
float NN::cosh(float x){return (exp(x)+exp(-x))/2;}
float NN::sinh(float x){return (exp(x)-exp(-x))/2;}
float NN::Sigmoid(float x){ return 1/(1+exp(-x)); }
float NN::DSigmoid(float x){ return Sigmoid(x) * (1-Sigmoid(x)); }
float NN::Logit(float x){ return log(x/(1-x)); }
float NN::Tanh(float x){ return tanhf(x); }
float NN::ATanh(float x){ return 0.5f * log((1 + x)/(1 - x)); }
float NN::DTanh(float x){
    float one = tanhf(x);
    return 1 - one*one;
}
float NN::LeakyRelu(float x){
    if(x > 0){ return x; }
    return 0.1*x;
}
float NN::DLeakyRelu(float x){
    if(x > 0){ return 1; }
    return 0.1;
}
float NN::Relu(float x){
    if(x > 0){ return x; }
    return 0;
}
float NN::DRelu(float x){
    if(x > 0){ return 1; }
    return 0;
}
float NN::Softmax(float x, int id){
    float exps = 0;
    for(int i = layerStarts[Nodes[id].layer]; i < Nodes.size(); i++){
        exps += exp(Nodes[i].z-maxoutput);
    }
    return exp(x-maxoutput)/exps;
}
float NN::Activation(float x, int id){
    if(LayerActivationNums[Nodes[id].layer] == 0){ return Sigmoid(x); }
    if(LayerActivationNums[Nodes[id].layer]==1){ return Relu(x); }
    if(LayerActivationNums[Nodes[id].layer]==2){ return LeakyRelu(x); }
    if(LayerActivationNums[Nodes[id].layer]==3){ return Tanh(x); }
    if(LayerActivationNums[Nodes[id].layer]==4){ return Softmax(x, id); }
    return 0;
}
float NN::DActivation(float x, int id){
    if(LayerActivationNums[Nodes[id].layer]==0){ return DSigmoid(x); }
    if(LayerActivationNums[Nodes[id].layer]==1){ return DRelu(x); }
    if(LayerActivationNums[Nodes[id].layer]==2){ return DLeakyRelu(x); }
    if(LayerActivationNums[Nodes[id].layer]==3){ return DTanh(x); }
    return 0;
}
float NN::InverseActivation(float x){
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

// loss functions and derivatives
float NN::logcosh(float x, float intended){
    return (log(NN::cosh(x-intended)));
}
float NN::Dlogcosh(float x, float intended){
    return (NN::sinh(x-intended)/NN::cosh(x-intended));
}
float NN::mse(float x, float intended){
    return ((intended - x) * (intended - x));
}
float NN::Dmse(float x, float intended){
    return -2 * (intended - x);
}
float NN::binarycrossentropy(float x, float intended){
    return -intended*log(x+1e-8) - (1-intended) * log(1 - x + 1e-8);
}
float NN::Dbinarycrossentropy(float x, float intended){
    return -intended/(x+1e-8) + (1-intended)/(1-x+1e-8);
}
float NN::CostFunction(float x, float intended){
    if(CostFunctionNum==0){ return mse(x, intended); }
    else if(CostFunctionNum==1){ return binarycrossentropy(x, intended); }
    else if(CostFunctionNum==2){ return logcosh(x, intended);}
    return 0;
}
float NN::DCostFunction(float x, float intended){
    if(CostFunctionNum==0){ return Dmse(x, intended); }
    else if(CostFunctionNum==1){ return Dbinarycrossentropy(x, intended); }
    else if(CostFunctionNum==2){ return Dlogcosh(x, intended);}
    return 0;
}

//network initializer, only ran without init true during backprop to allow thread creation
NN::NN(vector<int> _NNL, string _Function, string _CostFunctionStr, string _OptimizerStr, bool init){
    if(init == true){
        srand(time(0)); // used for random initialization of weights+biases
        Function = _Function;
        CostFunctionStr = _CostFunctionStr;
        OptimizerStr = _OptimizerStr;
        TL = _NNL.size();
        NNL = _NNL;
        int idcount = 0;
        //checks for function/optimizer/loss inputs
        if(Function=="Sigmoid"){ FunctionNum = 0; }
        else if(Function=="Relu"){ FunctionNum = 1; }
        else if(Function=="LeakyRelu"){ FunctionNum = 2; }
        else if(Function=="Tanh"){ FunctionNum = 3; }
        else{ std::cout << "Invalid activation function inputted"; return; }
        if(CostFunctionStr=="mse"){ CostFunctionNum = 0;}
        else if(CostFunctionStr=="binary crossentropy"){ CostFunctionNum = 1;}
        else if(CostFunctionStr=="logcosh"){ CostFunctionNum = 2; }
        else{ std::cout << "Invalid cost function inputted"; return; }
        if(OptimizerStr=="sgd"){ OptimizerNum = 0; }
        else if(OptimizerStr=="Msgd"){ OptimizerNum = 1; }
        else if(OptimizerStr=="adagrad"){ OptimizerNum = 2; }
        else if(OptimizerStr=="adadelta"){ OptimizerNum = 3; }
        else if(OptimizerStr=="adam"){ OptimizerNum = 4; }
        else{ std::cout << "Invalid optimizer inputted"; return; }
        for(int i = 0; i < TL; i++){
            LayerActivationStrings.push_back(Function);
            LayerActivationNums.push_back(FunctionNum);
        }
        std::default_random_engine generator;
        // initializing network values and lists
        for(int i = 0; i < NNL.size(); i++){
            for(int k = 0; k < NNL[i]; k++){
                Node nod;
                nod.id = idcount;
                nod.layer = i;
                nod.isInput = !(bool)i;
                nod.numInNodes = 0;
                if(!nod.isInput){
                    nod.numInNodes = NNL[i-1];
                    //random weights/bias initialization
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
                else{
                    nod.inWeights.push_back(0.0f);
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
                layerStarts.push_back(i); // populating layerstarts
                prevLayer = Nodes[i].layer;
            }
        }
        layerStarts.push_back(Nodes.size());
    }

}


// function used to change activation of a given layer to <activation>  Sigmoid//Relu//LeakyRelu//Tanh//Softmax
void NN::UpdateLayerActivation(NN& net, string activation, int layer){
    if(activation=="Sigmoid"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 0; }
    else if(activation=="Relu"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 1; }
    else if(activation=="LeakyRelu"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 2; }
    else if(activation=="Tanh"){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 3; }
    else if(activation=="Softmax" && layer==net.TL-1){ net.LayerActivationStrings[layer] = activation; net.LayerActivationNums[layer] = 4; }
    else{ std::cout << "Activation function inputted incorrectly" << "\n";}
}


// takes input data and returns network outputs after forward propagating // primary predict function
vector<float> NN::predict(NN& net, vector<float> Input){
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

//makes a prediction and then prints it << kinda useless
void NN::printPrediction(NN& net, vector<float> Input){
    vector<float> outs = predict(net, Input);
    for(int i = 0; i < outs.size(); i++){
        std::cout << Input[i] << " " << outs[i] << "\n";
    }
}

// train function for network, verbosity can be "verbose" or "silent"
void NN::Train(NN& net, vector<vector<float>> InData, vector<vector<float>> OutData, int batch, int epochs, float LR, string verbosity){
    net.lr = LR;
    net.inVal = InData;
    net.outVal = OutData;
    net.DataShuffle(net);
    float avgcost = 0;
    // set adjust rate (adadelta does not use LR)
    net.adjustRate = LR/batch;
    if(net.OptimizerNum==3||net.OptimizerNum==4){ net.adjustRate = LR; }
    // id of first output node
    int firstOutput = net.Nodes[net.Nodes.size() - 1].inNodes[net.Nodes[net.Nodes.size()-1].inNodes.size() - 1] + 1;
    int numThreads;
    // set threadnum (if less than batch, threadnum is used else batch size is used)
    if(threadNum<=batch){ numThreads = threadNum; }
    else{ numThreads = batch; }

    // derivatives of cost w.r.t. weights and biases
    vector<vector<float>> changesWeights; 
    vector<float> changesBiases;

    // filling variables.
    for(int NODE = 0; NODE < net.Nodes.size(); NODE++){
        changesBiases.push_back(0.0f);
        if(!net.updateVectorsInitialized){net.updateVectorBiases.push_back(0.0f); net.parameterUpdateBiases.push_back(0.0f);}
        changesWeights.push_back({});
        if(!net.updateVectorsInitialized){net.updateVectorWeights.push_back({}); net.parameterUpdateWeights.push_back({});}
        for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
            changesWeights[NODE].push_back(0.0f);
            if(!net.updateVectorsInitialized){net.updateVectorWeights[NODE].push_back(0.0f);net.parameterUpdateWeights[NODE].push_back(0.0f);}
        }
    }
    net.updateVectorsInitialized = true;

    // lambda used for multithreading, makes a copy of main network and uses copy to forward and backprop a data point and compute derivatives
    auto computeDerivatives = [&](int point){
        vector<float> baseDerivatives;
        for(int i = 0; i < net.Nodes.size(); i++){
            baseDerivatives.push_back(0.0f);
        }
        NN net1 (net.NNL, "LeakyRelu", "mse", "sgd", false);
        net.copyNet(net1, net);
        net1.inVal = {net.inVal[point]};
        net1.outVal = {net.outVal[point]};
        net1.loadValue(0, net1);
        net1.ForwardProp(net1);
        for(int i = 0; i < net1.outVal[0].size(); i++){
            avgcost += net1.CostFunction(net1.Nodes[firstOutput + i].a, net1.outVal[0][i]);
        }
        for(int NODE = net1.Nodes.size()-1; NODE >= 0; NODE--){
            if(net1.Nodes[NODE].layer==net.TL-1){
                if(net1.LayerActivationNums[net1.TL-1]!=4){
                    baseDerivatives[NODE] = net1.DCostFunction(net1.Nodes[NODE].a, net1.outVal[0][NODE-firstOutput]) * net1.DActivation(net1.Nodes[NODE].z, NODE);
                }
                else{
                    baseDerivatives[NODE] = 0;
                    for(int i = net1.layerStarts[net1.TL-1]; i < net1.Nodes.size(); i++){
                        baseDerivatives[NODE] += net1.DCostFunction(net1.Nodes[NODE].a, net1.outVal[0][NODE-firstOutput]) * net1.Softmax(net1.Nodes[i].z, i) * ((i == NODE) - net1.Softmax(net1.Nodes[NODE].z, NODE));
                    }
                }
            }
            else{
                baseDerivatives[NODE] = 0;
                for(int i = 0; i < net1.NNL[net.Nodes[NODE].layer + 1]; i++){
                    baseDerivatives[NODE]+=(baseDerivatives[net1.layerStarts[net1.Nodes[NODE].layer + 1] + i] * net1.Nodes[net1.layerStarts[net1.Nodes[NODE].layer + 1] + i].inWeights[NODE - net1.Nodes[net1.layerStarts[net1.Nodes[NODE].layer + 1] + i].inNodes[0]]);
                }
                baseDerivatives[NODE] *= DActivation(net1.Nodes[NODE].z, NODE);
            }
            changesBiases[NODE] += baseDerivatives[NODE] * BiasMult;
            if(net.OptimizerNum==2){ net.updateVectorBiases[NODE] += baseDerivatives[NODE] * BiasMult * baseDerivatives[NODE] * BiasMult; } 
            if(net.OptimizerNum==3){ net.updateVectorBiases[NODE] = net.momentumConst * net.updateVectorBiases[NODE] + (1-net.momentumConst)*baseDerivatives[NODE] * net.BiasMult * baseDerivatives[NODE] * net.BiasMult;}
            if(net.OptimizerNum==4){ net.updateVectorBiases[NODE] = net.adamB1*net.updateVectorBiases[NODE] + (1-net.adamB1)*baseDerivatives[NODE] * net.BiasMult; net.parameterUpdateBiases[NODE] = net.adamB2*net.parameterUpdateBiases[NODE] + (1-net.adamB2)*baseDerivatives[NODE] * net.BiasMult*baseDerivatives[NODE] * net.BiasMult;}
            if(net1.Nodes[NODE].layer>0){
                for(int i = 0; i < net1.Nodes[NODE].numInNodes; i++){
                    changesWeights[NODE][i] += baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a;
                    if(net.OptimizerNum==2){ net.updateVectorWeights[NODE][i] += baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a * baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a; } 
                    if(net.OptimizerNum==3){ net.updateVectorWeights[NODE][i] = net.momentumConst * net.updateVectorWeights[NODE][i] + (1-net.momentumConst) * baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a * baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a;}
                    if(net.OptimizerNum==4){ net.updateVectorWeights[NODE][i] = net.adamB1 * net.updateVectorWeights[NODE][i] + (1-net.adamB1) * baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a; net.parameterUpdateWeights[NODE][i] = net.adamB2 * net.parameterUpdateWeights[NODE][i] + (1-net.adamB2) * baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a* baseDerivatives[NODE] * net1.Nodes[net1.Nodes[NODE].inNodes[i]].a;}
                }
            }
        }
    };

    // main train loop
    for(int epoch = 0; epoch < epochs; epoch++){
        net.numEpochs++;
        // data is shuffled every epoch
        net.DataShuffle(net);
        vector<thread> threads;
        //as many threads as defined above/possible until entire dataset is cycled through network
        for(int point = 0; point < net.inVal.size(); point+=numThreads){
            for(int i = 0; i < numThreads && point+i < net.inVal.size(); i++){
                threads.push_back(thread{computeDerivatives, point+i});
            }
            for(int i = 0; i < numThreads && point+i < net.inVal.size(); i++){
                threads[i].join();
            }
            threads.clear();

            // updates parameters for each batch
            if(point/batch>=0 && point > 0){
                net.UpdateParams(net, changesWeights, changesBiases);
            }
        }
        
        // performs remaining updates if batch does not divide evenly into dataset size
        if(net.inVal.size()%batch!=0){
            net.UpdateParams(net, changesWeights, changesBiases);
        }

        // calculate and print cost
        std::cout << "cost: " << avgcost/net.inVal.size() << " Epoch: " << net.numEpochs << "\n";
        avgcost = 0;
    }
}
    
// light network copy used during backprop
void NN::copyNet(NN& dest, NN& source){
    dest.layerStarts = source.layerStarts;
    dest.TL = source.TL;
    dest.NNL = source.NNL;
    dest.LayerActivationNums = source.LayerActivationNums;
    dest.BiasMult = source.BiasMult;
    dest.Nodes = source.Nodes;
    dest.OptimizerNum = source.OptimizerNum;
    dest.CostFunctionNum = source.CostFunctionNum;
    dest.threadNum = source.threadNum;
}

// function to update network parameters after caclulating update vectors
void NN::UpdateParams(NN& net, vector<vector<float>> &WeightChanges, vector<float> &BiasChanges){
    //SGD
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
    // SGD momentum
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
    //adagrad
    if(net.OptimizerNum==2){
        net.adjustRate = net.lr;
        float update;
        for(int i = 0; i < net.NNL[0]; i++){
            update = net.adjustRate*BiasChanges[i]/sqrt(updateVectorBiases[i]+adagradConst);
            net.Nodes[i].bias-= update;
            BiasChanges[i] = 0;
        }
        for(int NODE = net.NNL[0]; NODE < net.Nodes.size(); NODE++){
            update = net.adjustRate*BiasChanges[NODE]/sqrt(updateVectorBiases[NODE]+adagradConst);
            net.Nodes[NODE].bias -= update;
            BiasChanges[NODE] = 0;
            for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                update = net.adjustRate * WeightChanges[NODE][i]/sqrt(net.updateVectorWeights[NODE][i] + adagradConst);
                net.Nodes[NODE].inWeights[i] -= update;
                WeightChanges[NODE][i] = 0;
            }
        }
    }
    //adadelta
    if(net.OptimizerNum==3){
        float update;
        for(int i = 0; i < net.NNL[0]; i++){
            update = net.adjustRate*sqrt(net.parameterUpdateBiases[i] + net.adagradConst)*BiasChanges[i]/sqrt(updateVectorBiases[i]+adagradConst);
            net.parameterUpdateBiases[i] = net.momentumConst * net.parameterUpdateBiases[i] + (1-net.momentumConst)*update*update;
            net.Nodes[i].bias-= update;
            BiasChanges[i] = 0;
        }
        for(int NODE = net.NNL[0]; NODE < net.Nodes.size(); NODE++){
            update = net.adjustRate*sqrt(net.parameterUpdateBiases[NODE] + net.adagradConst)*BiasChanges[NODE]/sqrt(updateVectorBiases[NODE]+adagradConst);
            net.parameterUpdateBiases[NODE] = net.momentumConst * net.parameterUpdateBiases[NODE] + (1-net.momentumConst)*update*update;
            net.Nodes[NODE].bias -= update;
            BiasChanges[NODE] = 0;
            for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                update = net.adjustRate*sqrt(net.parameterUpdateWeights[NODE][i] + net.adagradConst)*WeightChanges[NODE][i]/sqrt(net.updateVectorWeights[NODE][i] + adagradConst);
                net.parameterUpdateWeights[NODE][i] = net.momentumConst*net.parameterUpdateWeights[NODE][i] + (1-net.momentumConst)*update*update;
                net.Nodes[NODE].inWeights[i] -= update;
                WeightChanges[NODE][i] = 0;
            }
        }
    }
    //adam
    if(net.OptimizerNum==4){
        net.adamt+=1;
        float update;
        for(int i = 0; i < net.NNL[0]; i++){
            update = net.adjustRate *((net.updateVectorBiases[i]/(1-pow(net.adamB1, net.adamt)))/(sqrt(net.parameterUpdateBiases[i]/(1-pow(net.adamB2, net.adamt)))+net.adagradConst));
            net.Nodes[i].bias-= update;
            BiasChanges[i] = 0;
        }
        for(int NODE = net.NNL[0]; NODE < net.Nodes.size(); NODE++){
            update = net.adjustRate *((net.updateVectorBiases[NODE]/(1-pow(net.adamB1, net.adamt)))/(sqrt(net.parameterUpdateBiases[NODE]/(1-pow(net.adamB2, net.adamt)))+net.adagradConst));
            net.Nodes[NODE].bias -= update;
            BiasChanges[NODE] = 0;
            for(int i = 0; i < net.Nodes[NODE].numInNodes; i++){
                update = net.adjustRate *((net.updateVectorWeights[NODE][i]/(1-pow(net.adamB1, net.adamt)))/(sqrt(net.parameterUpdateWeights[NODE][i]/(1-pow(net.adamB2, net.adamt)))+net.adagradConst));;
                net.Nodes[NODE].inWeights[i] -= update;
                WeightChanges[NODE][i] = 0;
            }
        }
    }
}

// loads data point into network by point id
void NN::loadValue(int _datapiece, NN& net){
    net.datapiece = _datapiece;
    for(int i = 0; i < net.NNL[0]; i++){
        if(net.Nodes[i].isInput == true){
            net.Nodes[i].z = net.inVal[datapiece][i] + net.Nodes[i].bias*net.BiasMult;
            net.Nodes[i].a = Activation(net.Nodes[i].z, i);
        }
    }
}

// forward propagates one node
void NN::parrallelActivation(NN& net, NN::Node& nod){
    if(nod.isInput){ return; }
    nod.z = net.BiasMult * nod.bias;
    for(int i = 0; i < nod.numInNodes; i++){
        nod.z += nod.inWeights[i] * net.Nodes[i + nod.inNodes[0]].a;
    }
}

//forward props network, single threaded
void NN::ForwardProp(NN& net){
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
    net.maxoutput = 0;
    for(int i = net.layerStarts[net.TL-1]; i < net.Nodes.size(); i++){
        if(net.Nodes[i].z>=net.maxoutput){
            net.maxoutput = net.Nodes[i].z;
        }
    }
    for(int i = net.layerStarts[net.TL-1]; i < net.Nodes.size(); i++){
        net.Nodes[i].a = net.Activation(net.Nodes[i].z, i);
    }
}

//Shuffles dataset
void NN::DataShuffle(NN& net){
    vector<vector<float>> in = net.inVal;
    vector<vector<float>> out = net.outVal;
    vector<int> ids;
    for(int i = 0; i < in.size(); i++){ ids.push_back(i); }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    for(int i = 0; i < ids.size(); i++){
        net.inVal[i] = in[ids[i]];
        net.outVal[i] = out[ids[i]];
    }
}

// Saves network to a file
void NN::SaveNetwork(NN& net, std::string filename, std::string path){
    std::ofstream netFile;
    netFile.open(path+"/"+filename);
    netFile << net.TL << " ";
    for(auto i : net.NNL){
        netFile << i << " ";
    }
    netFile << "\n" << net.CostFunctionNum << " " << net.OptimizerNum << " " << net.BiasMult << " ";
    for(auto i : net.LayerActivationNums){
        netFile << i << " ";
    }
    netFile << "\n";
    for(auto i : net.Nodes){
        netFile << i.bias << " ";
    }
    netFile << "\n";
    for(auto i : net.updateVectorBiases){
        netFile << i << " ";
    }
    netFile << "\n";
    for(auto i : net.parameterUpdateBiases){
        netFile << i << " ";
    }
    netFile << "\n";
    for(auto i : net.Nodes){
        for(auto j : i.inWeights){
            netFile << j << " ";
        }
        netFile << ",";
    }
    netFile << "\n";
    for(auto i : net.updateVectorWeights){
        for(auto j : i){
            netFile << j << " ";
        }
        netFile << ",";
    }
    netFile << "\n";
    for(auto i : net.parameterUpdateWeights){
        for(auto j : i){
            netFile << j << " ";
        }
        netFile << ",";
    }
    netFile.close();    
}

// Load a network from a file
NN NN::LoadNetwork(std::string filename, std::string path){
    NN returnNet({1, 1}, "Sigmoid", "mse", "adam");
    int _TL = 0;
    vector<int> _NNL;
    int _costNum = 0;
    int _optNum = 0;
    vector<int> _layerAct;
    float _biasmult;
    fstream file;
    file.open((path+"/"+filename), ios::in);
    string line;
    int linecount = 0;
    while(getline(file, line)){
        if(line[line.size()-1]==',' || line[line.size()-1]==' ') line[line.size()-1] = '\0';
        string word;
        stringstream Line(line);
        int wordcount = 0;
        if(linecount==0){
            while(getline(Line, word, ' ')){
                if(wordcount==0) _TL = stoi(word);
                else{
                    _NNL.push_back(stoi(word));
                }
                wordcount++;
            }
        }
        else if(linecount==1){
            while(getline(Line, word, ' ')){
                
                if(wordcount==0) _costNum = stoi(word);
                else if(wordcount==1) _optNum = stoi(word);
                else if(wordcount==2) _biasmult = stof(word);
                else{
                    _layerAct.push_back(stoi(word));
                }
                wordcount++;
            }
            returnNet = NN(_NNL, "Sigmoid", "mse", "adam");
            returnNet.CostFunctionNum = _costNum;
            returnNet.OptimizerNum = _optNum;
            returnNet.LayerActivationNums = _layerAct;
            returnNet.BiasMult = _biasmult;
        }
        else if(linecount==2){
            while(getline(Line, word, ' ')){
                returnNet.Nodes[wordcount].bias = stof(word);
                wordcount++;
            }
        }
        else if(linecount==3){
            returnNet.updateVectorsInitialized = true;
            while(getline(Line, word, ' ')){
                returnNet.updateVectorBiases.push_back(stof(word));
                wordcount++;
            }
        }
        else if(linecount==4){
            while(getline(Line, word, ' ')){
                returnNet.parameterUpdateBiases.push_back(stof(word));
                wordcount++;
            }
        }
        else if(linecount==5){
            while(getline(Line, word, ',')){
                int subwordcount = 0;
                stringstream Word(word);
                string word1;
                while(getline(Word, word1, ' ')){
                    if(wordcount>=returnNet.layerStarts[1] && wordcount < returnNet.Nodes.size()) {
                        returnNet.Nodes[wordcount].inWeights[subwordcount] = stof(word1);
                    }
                    subwordcount++;
                }
                wordcount++;
            }
        }
        else if(linecount==6){
            while(getline(Line, word, ',')){
                returnNet.updateVectorWeights.push_back({});
                int subwordcount = 0;
                stringstream Word(word);
                string word1;
                while(getline(Word, word1, ' ')){
                    returnNet.updateVectorWeights[wordcount].push_back(0.0f);
                    if(wordcount>=returnNet.layerStarts[1]&& wordcount < returnNet.Nodes.size()) returnNet.updateVectorWeights[wordcount][subwordcount] = stof(word1);
                    subwordcount++;
                }
                wordcount++;
            }
        }
        else if(linecount==7){
            while(getline(Line, word, ',')){
                returnNet.parameterUpdateWeights.push_back({});
                int subwordcount = 0;
                stringstream Word(word);
                string word1;
                while(getline(Word, word1, ' ')){
                    returnNet.parameterUpdateWeights[wordcount].push_back(0.0f);
                    if(wordcount>=returnNet.layerStarts[1]&& wordcount < returnNet.Nodes.size()) returnNet.parameterUpdateWeights[wordcount][subwordcount] = stof(word1);
                    subwordcount++;
                }
                wordcount++;
            }
        }
        linecount++;
    }
    file.close();
    return returnNet;
}