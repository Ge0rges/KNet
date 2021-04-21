<div align="center">

# KNet

**A network that integrates plasticity and unsupervised learning in a way that models the brain's function.**

[![CodeFactor](https://www.codefactor.io/repository/github/ge0rges/knet/badge/master?s=2bd0ecc26d1a05b4bb8b1a7855a145c0cce0baf9)](https://www.codefactor.io/repository/github/ge0rges/knet/overview/backpropamine-based)

</div>

## State of the Effort
Thank you for taking interest in this project. After working on it for over a year, here is what we've learned. The architecture we came up initially did not work. For more information, check the [den-based](https://github.com/Ge0rges/KNet/tree/den-based) branch. 

This branch contains our latest effort which is based on modelling plasticity using the [backpropamine framework](https://github.com/uber-research/backpropamine) developed at Uber labs.

### Architecture Design
The idea behind our implementation is as follows. We set up an autoencoder network, and endow it with plasticity using backpropamine. 


### Testing
We intended to test our network by teaching it a widely different set of datasets. MNIST, Bananas Pictures and Car Pictures. We would then make sure that plasticity worked by verifying that the network learned all datasets adequatly without forgetting previosuly learned datasets. Then we would like to show that it encoded "meaning" properly by showing it a [banana-car](https://i.ytimg.com/vi/_9Nm_aI_7hc/maxresdefault.jpg) and showing that it recognizes the presence of a banana as well as a car. 

## Bigger Picture
My research interest is to model the functions of the brain by integrating existing machine learning models together.
Model the entire process by which humans have come to exist using various types of neural networks, and a computational base.

  Energy + Luck + Matter -> Simple cells (+ Time + Natural Selection) -> Humans
  Energy + Human + Matter -> Simple Computation/NN (+ Genetic Algorithm/Gradient Descent + Time) -> Candidate Nets

We take humans as an example of general intelligence since we are certain of it.
Humans exhibit:
- Encoding specificity
- Plasticity
- Long/Short Term Memory (imperfect)
- Unsupervised Learning
- Reinforcement Learning
- Highly efficient
- Classification
- Efficient Abstraction
- Relevance Realization
- General Intelligence
- Consciousness

Candidate Networks:
1. Should exhibit many human attributes, many of which correspond to existing NNs:
  - Recurrent LSTM (Memory)
  - Wake-Sleep (Unsupervised Learning)
  - Backpropamine/DEN (Plasticity)
  - Reinforcement Learning
2. Will learn from it's parent (previous generation of genetic algorithm) and the world,
    much like humans learn from their parents and the world.
3. Will have simple computational bases (genes = code, simple computation = cells that react)
4. It is possible that early generations be "simpler" nets the same way animals appear "simpler" than humans.


*This repository aims to integrate unsupervised learning, plasticity and energy constrained trainning*

## Software 
### How to Run
1. Install the dependencies below.
2. Pick the experiment you'd like to run, find it's corresponding run_experiment.py file.
3. Run prepare_experiment()
4. Optionally, run find_hypers() to get a good set of hyperparameters.
5. Run train_model(). Parameters are defined inside the function.


### Dependencies
- Python 3.8
- numpy 1.20.2
- Pillow 7.1.0 or Pillow-SIMD 7.1.0
    - For pillow-simd you will need a SSE4.1 capable CPU and `sudo apt-get install libjpeg-dev zlib1g-dev`
- torch 1.8.1

## Style Guide
We use the industry standard [PEP8].

[PEP8]: <https://pep8.org>

## Resources
Below a compilation of resources we've used along the way, including academic papers, tutorials, etc.
### Papers
- [The wake-sleep algorithm for unsupervised neural networks](https://www.cs.toronto.edu/~hinton/absps/ws.pdf)
- [Lifelong Learning with Dynamically Expandable Networks](https://openreview.net/pdf?id=Sk7KsfW0-) with [code](https://github.com/jaehong-yoon93/DEN)
- [Measuring abstract reasoning in neural networks](https://arxiv.org/pdf/1807.04225.pdf)
- [Backpropamine: differentiable neuromodulated plasticity](https://openreview.net/pdf?id=r1lrAiA5Ym)
- [Differentiable plasticity: training plastic neural networks with backpropagation](http://proceedings.mlr.press/v80/miconi18a/miconi18a.pdf)

### Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- [Car 1](https://archive.ics.uci.edu/ml/datasets/Automobile)
- [Stanford Car](https://www.kaggle.com/jessicali9530/stanford-cars-dataset)
- [Fruit 360](https://www.kaggle.com/moltean/fruits) (organized by subfolders)
- [Mendeley Banana](https://data.mendeley.com/datasets/zk3tkxndjw/2)
- [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)

## Thanks
Thanks to Prof. John Vervaeke for guidance, thoughts and wisdom.

Thanks to Prof. Steve Mann for providing some computational resources.

Thanks to mikacho for guidance on the resource constraining function.
