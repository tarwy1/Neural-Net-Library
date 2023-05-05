
#include "neuralnet/neuralnet.hpp"
#include "./mnist/mnist_reader.hpp"
#include <iostream>

using namespace std;

int main(){
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./mnist/");
    
    NN Network({784, 32, 16, 10}, "LeakyRelu", "binary crossentropy", "adam");
    
    Network.UpdateLayerActivation(Network, "Sigmoid", 3);

    vector<vector<float>> inData;
    vector<vector<float>> outData;

    for(int i = 0; i < 10000; i++){
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
    std::cout << "training\n";
    ofstream myfile;
    myfile.open ("outData.csv");
    for(int ep = 1; ep<20; ep++){
        Network.Train(Network, Network.inVal, Network.outVal, 32, 1, 0.01f);
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
        myfile << ep << ","<<  totalRight/100 << "\n";
        std::cout << totalRight/100 << "\n";
    }
    Network.SaveNetwork(Network, "kekw.txt", "C:/NN");
    
    myfile.close();
    
    Network = Network.LoadNetwork("kekw.txt", "C:/NN");

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
    std::cout << "\n" << totalRight/100 << "\n";
    return 0;
}
