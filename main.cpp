
#include "neuralnet/neuralnet.hpp"
#include "./mnist/mnist_reader.hpp"
#include <iostream>

using namespace std;

/*
This program serves as a basic MNIST training example using the network
A very simple network structure and only 1/6 of the total MNIST dataset is used here for training speed
This model should reach ~90% accuracy on the testing dataset
*/

int main(){
    // Importing the MNIST dataset
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./mnist/");
    
    // Creating the network
    // This network uses leaky relu in all layers but the output,
    // binary crossentropy loss and the ADAM optimiser
    NN Network({784, 32, 10}, "LeakyRelu", "binary crossentropy", "adam");
    
    Network.UpdateLayerActivation(Network, "Sigmoid", 2);

    // Generating training and testing datasets
    vector<vector<float>> inData;
    vector<vector<float>> outData;
    for(int i = 0; i < 10000; i++){
        // Iterate through the MNIST dataset and append the values into inData and outData
        inData.push_back({});
        for(int j = 0; j < 784; j++){
            // Mnist images are 28*28 which means 784 total input nodes
            // They are also greyscale with 8-bit values. hence the /255
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

    // Generating testing data from MNIST the same way as above
    vector<vector<float>> testingData;
    vector<float> testingAnswers;
    for(int i = 0; i < 10000; i++){
        testingData.push_back({});
        for(int j = 0; j < 784; j++){
            testingData[i].push_back((float)dataset.test_images[i][j]/255);
        }
        testingAnswers.push_back((int)dataset.test_labels[i]);
    }
    
    // Main training loop
    std::cout << "training\n";
    // This is configured to 10 epochs using a for loop,in order to calculate accuracy at each epoch, 
    // This could be done directly in the Train function with the epochs parameter
    for(int ep = 1; ep<=10; ep++){
        // Train function (batch size 128, 1 epoch (epochs done with for loop), 0.01 learning rate)
        Network.Train(Network, inData, outData, 128, 1, 0.01f);
        
        // Calculating accuracy for that epoch
        // By using the testing data and calculating the % of them that the network predicts correctly
        // Note i am incrementing by 10 so as to reduce time taken for these checks during training
        float totalRight = 0;
        for(int i = 0; i < 10000; i+=10){
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
        std::cout << "Accuracy: " << totalRight/10 << "%\n";

        // Saving network on every epoch (can be removed for quicker training networks)
        // This allows you to ctrl+C stop the code execution at any time and keep the network
        Network.SaveNetwork(Network, "NetworkFile.txt", "C:/NN");
    }

    // Finding final testing data accuracy by the same method as during training
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
    std::cout << "\nFinal Accuracy: " << totalRight/100 << "%\n";
    return 0;
}
