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
        - [Binary Crossentropy](#binary-crossentropy)
        - [Log-Cosh](#log-cosh)
    - [Optimizers](#optimizers)
        - [stochastic gradient descent](#stochastic-gradient-descent)
        - [SGD Momentum](#sgd-momentum)
        - [AdaGrad](#adagrad)
        - [AdaDelta](#adadelta)
        - [Adam](#adam)
- [Functional Principals and Mathematics](#functional-principals-and-mathematics)
    - [Overall Network Structure](#overall-network-structure)
    - [Calculating Backprop Derivatives](#calculating-backprop-derivatives)
    - [CPU Multithreading](#cpu-multithreading)
- [Using the Library](#using-the-library)

# Project Overview:
This project was intended as a proof of concept and as a learning opportunity for how neural networks function and how libraries such as [Tensorflow](https://www.tensorflow.org/) are made. The core component of this project is a library which can be imported that provides tools for the creating and use of densely connected neural networks. The library allows for the creation of networks of arbitrary size, number of layers, and with user selected parameters such as activation function and optimizer.\
\
Throughout the development of this library, I found that online documentation of neural network functionality is lacking overall and as such, many elements of the design and core functionality of the library had to be theorised and developed from first principles without external help. 

## Features:
The library allows for the creation, saving and loading of densely connected neural networks with arbitrary number of layers and nodes in each layer.
### Activation Functions:
The activation function of each layer can be set as desired to any of the below functions.
#### Sigmoid:
Logistic activation function which maps the input between 0 and 1.\
$$S(x) = \frac{1}{1-e^{-x}}$$
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



