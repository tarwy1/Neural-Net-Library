
#include "neuralnet/neuralnet.hpp"
#include <iostream>

using namespace std;

int main(){
    //example code using network to appproximate sin(x)+1 function in domain -10 < x < 10

    NN Network({1, 10, 10, 1}, "LeakyRelu", "logcosh", "adadelta");

    for(float i = 0; i < 20; i += 0.0008){
        Network.inVal.push_back({i});
        Network.outVal.push_back({(sin(i-10)+1)});
    }
    Network.Train(Network, Network.inVal, Network.outVal, 1, 8, 1.0f);

    vector<float> TestingData;
    vector<float> TestingAnswers;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(0.0f, 20.0f);
    for(int i = 0; i < 10000; i++){
        TestingData.push_back(dist(generator));
        float one = TestingData[i];
        //std::cout << TestingData[i] << " " << one << "\n";
        TestingAnswers.push_back(sin(one-10)+1);
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
