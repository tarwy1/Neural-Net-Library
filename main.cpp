
#include "neuralnet/neuralnet.hpp"
#include <iostream>

using namespace std;

int main(){
    //example code using network to appproximate sin(x)+1 function in domain -10 < x < 10

    NN Network({1, 10, 10, 1}, "LeakyRelu", "logcosh", "adam");

    for(float i = 0; i < 5; i += 0.001){
        Network.inVal.push_back({i});
        Network.outVal.push_back({(atan(exp(i)))});
    }
    Network.Train(Network, Network.inVal, Network.outVal, 1, 30, 0.01f);

    vector<float> TestingData;
    vector<float> TestingAnswers;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(0.0f, 5.0f);
    for(int i = 0; i < 10000; i++){
        TestingData.push_back(dist(generator));
        float one = TestingData[i];
        //std::cout << TestingData[i] << " " << one << "\n";
        TestingAnswers.push_back(atan(exp(one)));
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
