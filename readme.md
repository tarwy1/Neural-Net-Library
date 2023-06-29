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
- [Functional Principles and Mathematics](#functional-principles-and-mathematics)
    - [Overall Network Structure](#overall-network-structure)
    - [Computing Backprop Derivatives](#computing-backprop-derivatives)
    - [CPU Multithreading](#cpu-multithreading)
- [Using the Library](#using-the-library)

# Project Overview:
This project was intended as a proof of concept and as a learning opportunity for how neural networks function and how libraries such as [Tensorflow](https://www.tensorflow.org/) are made. The core component of this project is a library which can be imported that provides tools for the creating and use of densely connected neural networks. The library allows for the creation of networks of arbitrary size, number of layers, with user selected parameters such as activation function and optimizer.\
\
Throughout the development of this library, I found that online documentation of neural network functionality is lacking overall, as such, many elements of the design and core functionality of the library had to be theorised and developed from first principles without external help. \
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
This library allows for the use of any of the below optimizers with both a variable batch size and learning rate, other hyperparameters such as the momentum decay constant can also be modified as public variables in the neural network class. For each of the following equations, \
$\theta =$ network parameters, η = learning rate, $L(\theta) =$ network loss as a function of $\theta$ and $\nabla_{\theta} L(\theta) =$ a partial derivative of the loss with respect to theta, for brevity this will sometimes be written $g_t$. \
\
Most of the theory and equations for the optimizers used in my network were sourced from [this article](https://www.ruder.io/optimizing-gradient-descent/) on [ruder.io](https://www.ruder.io/) which proved to be one of the best resources I could find on optimizing gradient descent.
#### Stochastic Gradient Descent:
This optimizer functions by constantly overshooting the local minima but overtime converging on it simply by taking finding the gradient of the loss w.r.t each parameter, multiplying it by the constant learning rate and subtracting from the parameter. This effectively moves the parameter in a direction so as to minimise the value of loss.
$$\theta = \theta - η \cdot \nabla_{\theta} L(\theta)$$
#### SGD Momentum:
Momentum aims to counteract the tendency of SGD to struggle when approaching closely to local minima by first calculating an update vector $v_t$ which is dependant on the previous by a constant. This reduces the tendency of SGD to 'bounce from one side of the ravine to the other' if the local minima is thought of as a ravine.
$$v_t = \gamma v_{t-1} + η \cdot \nabla_{\theta} L(\theta)$$
$$\theta = \theta - v_t$$
where $\gamma$ is a constant usually set to approximately 0.9.
#### AdaGrad:
This optimizer allows for the dynamic adjustment of the learning rate for each paramter, e.g. for a parameter which consistently has a large impact on the loss, smaller updates will be made and for parameters which have less impact on the loss, larger updates. This is done by keeping a sum of previous squared gradients and using it in the update rule as follows:
$$G_t = \sum_{t=0}^{t} g_t^2$$
$$\theta = \theta - \frac{η}{\sqrt{G_t + \epsilon}} \cdot g_t$$
The main weakness of this optimizer however is the growing denominator in the update terms which is always increasing, this leads to quickly decaying gradients at which point the network effectively stagnates.
#### AdaDelta
AdaDelta is a modification built upon AdaGrad which seeks to relieve the decaying update vector problem by only keeping track of more recent past square gradients in the $G_t$ term. This is done not by storing n past gradients but by redifining $G_t$ as a decaying sum of past square gradients as follows:
$$G_t = \gamma G_{t-1} + (1-\gamma) g_t^2$$
Where $\gamma$ is a similar value to with momentum, approximately 0.9. \
\
Adadelta differs from AdaGrad in the update rule aswell however. It also keeps a decaying sum of past square update vectors which can be called $E_t$ and is defined as follows:
$$E_t = \gamma E_{t-1} + (1-\gamma) v_t^2$$
Where $v_t$ stems from the update rule:
$$\theta = \theta - v_t$$
and is now defined as:
$$v_t = \frac{\sqrt{E_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} \cdot g_t$$
The cyclic dependency of $E_t$ on $v_t$ is fixed by using $E_{t-1}$ in the definition of $v_t$ as shown above.. \
In full the update rule for AdaDelta is:
$$\theta = \theta - \frac{\sqrt{E_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} \cdot g_t$$
As you can see, we have eliminated the learning rate from the equation entirely, with it being replaced by the decaying update vectors term.
#### Adam:
Adam or Adaptive Moment Estimation works similarly to AdaDelta in that it also keeps a decaying sum of past squared gradients $G_t$ but it differs by also keeping a decaying sum of average past gradients $M_t$ as follows:
$$M_t = \beta_1 M_{t-1} + (1-\beta_1) g_t $$
$$G_t = \beta_2 G_{t-1} + (1-\beta_2) g_t^2$$
The authors of Adam however, noted that these decaying sums are biased towards zero and in order to relieve that bias, they calculate bias corrected sums $\hat{G_t}$ and $\hat{M_t}$
$$\hat{M_t} = \frac{M_t}{1-\beta_1^t}$$
$$\hat{G_t} = \frac{G_t}{1-\beta_2^t}$$
Although it does not look like it (and this confused me for a long time... ) the t exponent on $\beta$ in the bias corrected sums is intentional and serves to force the adam update vector to continue to make significant changes even after many epochs. \
We then get the final update rule for Adam as follows:
$$\theta = \theta - \frac{η}{\sqrt{\hat{G_t}} + \epsilon} \cdot \hat{M_t} $$
Where values of 0.9, 0.999 and $10^{-8}$ are typically used for $\beta_1$, $\beta_2$ and $\epsilon$ respectively.
# Functional Principles and Mathematics
I based this neural network library on the commonly accepted and used structure for a densely connected neural network as shown: \
<img src="https://github.com/tarwy1/Neural-Net-Library/assets/38536921/7ded112c-941a-40d7-886d-28a77d29da37"  width="50%" height="50%"> \
This basic model consists of an input layer, an output layer and one or more hidden layers. \
\
In simple terms, each node is connected to every node in the next layer by a weight and the activation or value of each node can be computed by multiplying the activation of the node in the previous layer by the corrresponding weight connecting it. \
\
Each node also has a bias which is added onto this weighted sum and an activation function which is applied to the sum of the weighted sum and bias to give the final activation of the node. This is represented mathematically below: 

$$ z^L_n = \sum_{i=0}^i (w^L_{ni} \cdot a^{L-1}_i ) + B_n^L \cdot B_0$$ 

$$ a^L_n = \sigma(z^L_n) $$ 

Where $a^L_n$ is the activation of node n in layer L (output layer) after the activation function, $W^L_{ni}$ is the weight connecting node n in layer L to node i in layer L-1, $B^L_n$ is the bias of node n in layer L and $B_0$ is a universal constant used as a hyperparameter to adjust the influence of biases on the network (typically 1.0). \
\
The weights are randomly initialized using a general rule and are then updated by forward propagating a data sample through the network by the above rule, then a cost or loss is calculated using a function of the intended output values and the value obtained from forward propagation, this cost effectively represents how inaccurate the network was for that sample. \
\
The derivative of that loss function is the found with respect to each parameter in the network (weights and biases) and is used to update the weights according to an update rule or 'optimizer'. \
\
The next sample can then be propagated through the network and the process repeats. Over time this allows the network to 'learn' how to produce the desired output from an input.
## Overall Network Structure:
Much of the network structure for example, how nodes are stored in code was based on some of the information in [this paper](https://www.researchgate.net/publication/341310964_GPU_Acceleration_of_Sparse_Neural_Networks) which describes a method of storing each node as a structure as shown: 

<img src="https://github.com/tarwy1/Neural-Net-Library/assets/38536921/b6aa97fd-ccdc-4bae-ab40-33b259875df3"  width="50%" height="50%">

Each node is a structure which contains its id, layer (input = 0), the number of nodes in the previous layer, a boolean for if the node is an input, a list of the ids of nodes in the previous layer, a list of weight values connecting to corresponding nodes in the previous layer, the two activation variables, and the bias value of the node. 

When the train function is ran, the training data and labels are loading into the network, shuffled and then for each epoch, the dataset is reshuffled, the network is forward-propagated, back-propagated to find the loss derivatives w.r.t each parameter, and then each paramter is updated according to the update rule.














