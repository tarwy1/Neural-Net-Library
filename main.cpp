#include <mnist/mnist_reader.hpp>
#include "neuralnet/neuralnet.hpp"

using namespace std;

int main(){

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./mnist-master/mnist-master/");

    NN Network({1, 10, 10, 1}, "LeakyRelu", "mse", "adagrad");

    //Network.UpdateLayerActivation(Network, "Sigmoid", 3);

    // vector<vector<float>> inData;
    // vector<vector<float>> outData;

    // for(int i = 0; i < 60000; i++){
    //     inData.push_back({});
    //     for(int j = 0; j < 784; j++){
    //         inData[i].push_back(((float)dataset.training_images[i][j])/255);
    //     }
    //     outData.push_back({});
    //     for(int j = 0; j < 10; j++){
    //         if(j==(int)dataset.training_labels[i]){
    //             outData[i].push_back(1.0f);
    //         }
    //         else{
    //             outData[i].push_back(0.0f);
    //         }
    //     }
    // }
    
    // Network.inVal = inData;
    // Network.outVal = outData;

    for(float i = -10; i < 10; i += 0.001){
        Network.inVal.push_back({i});
        Network.outVal.push_back({(sin(i)+1)});
    }
    Network.Train(Network, Network.inVal, Network.outVal, 32, 50, 0.1f);

    // vector<vector<float>> testingData;
    // vector<float> testingAnswers;
    // for(int i = 0; i < 10000; i++){
    //     testingData.push_back({});
    //     for(int j = 0; j < 784; j++){
    //         testingData[i].push_back((float)dataset.test_images[i][j]/255);
    //     }
    //     testingAnswers.push_back((int)dataset.test_labels[i]);
    // }
    // float totalRight = 0;
    // for(int i = 0; i < 10000; i++){
    //     vector<float> prediction = Network.predict(Network, testingData[i]);
    //     int maxindex = 0;
    //     for(int i = 0; i < 10; i++){
    //         if(prediction[i] >prediction[maxindex]){
    //             maxindex = i;
    //         }
    //     }
    //     if((int)maxindex==(int)testingAnswers[i]){
    //         totalRight++;
    //     }
    // }
    // std::cout << totalRight/10000*100 << "\n";

    vector<float> TestingData;
    vector<float> TestingAnswers;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for(int i = 0; i < 10000; i++){
        TestingData.push_back(dist(generator));
        float one = TestingData[i];
        //std::cout << TestingData[i] << " " << one << "\n";
        TestingAnswers.push_back(sin(one)+1);
    }
    ofstream myfile;
    myfile.open ("outData.csv");
    for(int i = 0; i < TestingData.size(); i++){
        std::cout << TestingData[i] << " ";
        std::cout << Network.predict(Network, {TestingData[i]})[0] << " ";
        std::cout << TestingAnswers[i] << "\n";
        myfile << TestingData[i] << ",";
        myfile << Network.predict(Network, {TestingData[i]})[0] << ",";
        myfile << TestingAnswers[i] << "\n";
    }

    
    myfile.close();
    return 0;
}
