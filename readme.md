# **Table of contents**
- [Table of contents](#table-of-contents)
- [Project Overview](#project-overview)
  - [Features](#features)
    - [Activation Functions](#activation-functions)
        - [Sigmoid](#sigmoid)
        - [RELU](#relu)
        - [Leaky RELU](#leaky-relu)
        - [Tanh](#tanh)
        - [Softmax](#softmax)
    - [Cost Functions](#cost-functions)
        - [Mean Squared Error](#mean-squared-error)
        - [Binary Cross entropy](#binary-cross-entropy)
        - [Log-Cosh](#log-cosh)
    - [Optimizers](#optimizers)
        - [stochastic gradient descent](#stochastic-gradient-descent)
        - [SGD Momentum](#sgd-momentum)
        - [AdaGrad](#adagrad)
        - [AdaDelta](#adadelta)
        - [Adam](#adam)
- [Functional Principals and Mathematics](#functional-principals-and-mathematics)
    - [Overall Network Structure](#overall-network-structure)
    - [Computing Backprop Derivatives](#computing-backprop-derivatives)
    - [CPU Multithreading](#cpu-multithreading)
- [Using the Library](#using-the-library)

# Project Overview:
This project was intended as a proof of concept and as a learning opportunity for how neural networks function and how libraries such as [Tensorflow](https://www.tensorflow.org/) are made. The core component of this project is a library which can be imported that provides tools for the creating and use of densely connected neural networks. The library allows for the creation of networks of arbitrary size, number of layers, and with user selected parameters such as activation function and optimizer.\
\
Throughout the development of this library, I found that online documentation of neural network functionality is lacking overall and as such, many elements of the design and core functionality of the library had to be theorised and developed from first principles without external help. \
\
All functional elements of the project were designed and programmed by me, with some assistance from [Th3T3chn0G1t](https://github.com/Th3T3chn0G1t) with converting the code into an importable library.

## Features:
The library allows for the creation, saving and loading of densely connected neural networks with arbitrary number of layers and nodes in each layer.
### Activation Functions:
The activation function of each layer can be set as desired to any of the below functions.
#### Sigmoid:
Logistic activation function which maps the input between 0 and 1.\
$$S(x) = \frac{1}{1+e^{-x}}$$
#### RELU:
Rectified linear activation function. Defined by a piecewise function as shown below:
$$f(x)=0  \quad for \quad x < 0$$
$$f(x)=x \quad  for \quad x ≥ 0$$
#### Leaky Relu:
RELU with a constant multiplier for negative values to prevent dead nodes.
$$f(x)=0.1x  \quad for \quad x < 0$$
$$f(x)=x \quad  for \quad x ≥ 0$$
#### Tanh:
Maps the input from -1 to 1, allowing greater flexibility than sigmoid in some tasks.
$$Tanh(x) = \frac{e^x – e^{-x}}{e^x + e^{-x}}$$
#### Softmax:
Takes every node in the layer and returns probability values for each node, which sum to 1, used in classification tasks.
$$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
To find the softmax value of a given node, divide the exp of that node by the sum of exp of all nodes in the layer.
### Cost Functions:
The library allows for the selection of a cost (loss) function at the time of creation of the network from the following list\
where $x_i$ is defined as the output of node i and $y_i$ is defined as the "correct value" or desired value of the node
#### Mean Squared Error:
This is the most basic neural network cost function defined as the sum of square errors of each output node:
$$MSE(\vec{x}) = \sum_{i=0}^{N} (x_i - y_i)^2$$
#### Binary Cross entropy:
This is a probabilistic loss function intended to be used in classification problems (e.g. MNIST) and should be used with the Sigmoid activation function in the output layer.
$$Loss(\vec{x}) = -\frac{1}{N} \sum_{i=0}^{N} \left [ y_i log(x_i) + (1-y_i)log(1-x_i)\right ]$$
In my code, this is also implemented with a $1*10^{-8}$ term in both logs to prevent taking the log of 0
#### Log-Cosh:
This loss function is intended as a direct replacement or improvement on the MSE loss function, it aims to fix the tendency of MSE to bounce around a local minima with larger learning rates.
$$loss(\vec{x}) = \sum_{i=0}^{N} log(cosh(x_i-y_i))$$
### Optimizers:
This library allows for the use of any of the below optimizers with both a variable batch size and learning rate, other hyperparameters such as the momentum decay constant can also be modified as public variables in the neural network class. For each of the following equations, $\theta =$ network parameters, η = learning rate.
#### Stochastic Gradient Descent:
This optimizer functions by constantly overshooting the local minima but overtime converging on it




