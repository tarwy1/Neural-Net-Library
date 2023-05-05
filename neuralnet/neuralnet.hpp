#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <random>
#include <ctime>
#include <cmath>
#include <thread>
#include <sstream>

class NN{
    public:
        // activation functions and derivatives and inverses(for some)
        float cosh(float x);
        float sinh(float x);
        float Sigmoid(float x);
        float DSigmoid(float x);
        float Logit(float x);
        float Tanh(float x);
        float ATanh(float x);
        float DTanh(float x);
        float LeakyRelu(float x);
        float DLeakyRelu(float x);
        float Relu(float x);
        float DRelu(float x);
        float Softmax(float x, int id);
        float Activation(float x, int id);
        float DActivation(float x, int id);
        float InverseActivation(float x);

        // loss functions and derivatives
        float logcosh(float x, float intended);
        float Dlogcosh(float x, float intended);
        float mse(float x, float intended);
        float Dmse(float x, float intended);
        float binarycrossentropy(float x, float intended);
        float Dbinarycrossentropy(float x, float intended);
        float CostFunction(float x, float intended);
        float DCostFunction(float x, float intended);

        struct Node{
            int id; // node id
            int layer; // layer its in,  0 is the input layer
            int numInNodes; // number of nodes in the previous layer
            bool isInput; // is in input layer
            std::vector<int> inNodes; // list of the id's of nodes in previous layer
            std::vector<float> inWeights; // list of values of the weights between nodes in previous layer
            float z; // z  activation before function
            float a; // a  activation after activation function
            float bias; // bias
        };

        int TL; // total number of layers
        std::vector<int> NNL;  // list containing number of nodes in each layer
        std::vector<Node> Nodes; // list of nodes including all values
        std::vector<std::vector<float>> inVal; // input data list
        std::vector<std::vector<float>> outVal; // output data/data labels list
        int datapiece; // int representing which data point is being processed
        std::string Function; // activation function std::string used for primary activation function applied to all layers   Sigmoid//Relu//LeakyRelu//Tanh//Softmax
        int FunctionNum = 0; // activation function int
        std::string CostFunctionStr; // cost function std::string
        int CostFunctionNum = 0; // cost function int
        float BiasMult = 1.0f; // bias multiplier (when calculating activation, bias is multiplied by this for all nodes)
        std::string OptimizerStr; // optimizer std::string
        int OptimizerNum = 0; // optimizer int
        std::vector<int> layerStarts; // list of starting node ids for each layer
        float momentumConst = 0.9f; // constant γ used in algorithms such as momentum and adadelta
        std::vector<std::vector<float>> updateVectorWeights; // contains update information for weights during backprop
        std::vector<float> updateVectorBiases; // contains update information for biases during backprop
        float adjustRate = 0.0001f; // learning rate divided by batch size, calculated during backprop, initialized to a base value
        bool updateVectorsInitialized = false; // check bool during initialization for backprop
        std::vector<std::string> LayerActivationStrings; // std::string for activation function of each layer
        std::vector<int> LayerActivationNums; // int for activation function of each layer
        float maxoutput = 0.0f; // max output variable used for softmax calculation
        float adagradConst = 1e-5; // adagrad constant
        int threadNum = 64; // number of threads created during backprop, determined then
        float lr; // learning rate
        // lists used to track weighted averages for adagrad, delta etc
        std::vector<std::vector<float>> parameterUpdateWeights; 
        std::vector<float> parameterUpdateBiases;
        float adamB1 = 0.9f;
        float adamB2 = 0.999f;
        int adamt = 0;
        int numEpochs = 0;

        //network initializer, only ran without init true during backprop to allow thread creation
        NN(std::vector<int> _NNL, std::string _Function, std::string _CostFunctionStr, std::string _OptimizerStr, bool init = true);

        // function used to change activation of a given layer to <activation>  Sigmoid//Relu//LeakyRelu//Tanh//Softmax
        void UpdateLayerActivation(NN& net, std::string activation, int layer);

        // takes input data and returns network outputs after forward propagating // primary predict function
        std::vector<float> predict(NN& net, std::vector<float> Input);

        //makes a prediction and then prints it << kinda useless
        void printPrediction(NN& net, std::vector<float> Input);

        // train function for network, verbosity can be "verbose" or "silent"
        void Train(NN& net, std::vector<std::vector<float>> InData, std::vector<std::vector<float>> OutData, int batch, int epochs, float LR, std::string verbosity = "verbose");
    
        // saves the network configuration and parameters to a file
        void SaveNetwork(NN& net, std::string filename, std::string path);

        // loads a network from the file
        NN LoadNetwork(std::string filename, std::string path);


    private:
        // light network copy used during backprop
        void copyNet(NN& dest, NN& source);
        // function to update network parameters after caclulating update vectors
        void UpdateParams(NN& net, std::vector<std::vector<float>> &WeightChanges, std::vector<float> &BiasChanges);

        // loads data point into network by point id
        void loadValue(int _datapiece, NN& net);

        // forward propagates one node
        void parrallelActivation(NN& net, NN::Node& nod);
        
        //forward props network, single threaded
        void ForwardProp(NN& net);

        //Shuffles dataset
        void DataShuffle(NN& net);
        
};
