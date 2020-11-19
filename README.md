<div align="center">

# KNet

**A network that integrates plasticity and unsupervised learning in a way that models the brain's function.**

[![CodeFactor](https://www.codefactor.io/repository/github/ge0rges/knet/badge/master?s=2bd0ecc26d1a05b4bb8b1a7855a145c0cce0baf9)](https://www.codefactor.io/repository/github/ge0rges/knet/overview/master)

</div>

## State of the Effort
Thank you for taking interest in this project. After working on it for over a year, here is what we've learned. The arhcitecture we came up with does not work. We can't guarrantee it's not due to a bug or mathemtical error, but we have stopped pursuing research and development of it. Instead we pivoted to working on the [Plastic Support Structure](https://github.com/Ge0rges/Plastic-Support-Structure). 

We have attempted different variations of the architecture, adn this repo's current state is as it was before we pivoted. 

We hope that by open sourcing this code, someone can highlight our errors or continue our work. Ge0rges is interested in pursuing this idea through a seperate research effort focused on some of the same core ideas but in a different implementation and architecture. Feel free to contact him. Please feel free to open an issue if you'd like more details on what we tried or about the project in general. 

### Architecture Design
The idea behind our implementation is as follows. We set up an autoencoder network, and endow it with plasticity based on a modified DEN algortihm. We add a feed forward network on top of the autoencoder called the action network. The action network takes as inputs the outputs of the middle layer of the autoencoder, and outputs the category. The action network is equally plastic. 

The idea behind this design is inspired from our brain. We suppose that the autoencoder will encode meaning by compressing the data into it's most invariant forms. In essence, the autoencoder attempts to replicate meaning by way of it's core invairant layer (middle layer) within our brain, and unsupervised learning by way of it's training method. The logic behind the action network is that in order to categorize something, we take an input, compare it to our "meaning database", then produce an output. That is what the action network aims to replicate, by taking the processed input at the core-invariant label then classifying it. 

Training occurs in a modified way. The autoencoder is trained classicaly with a mean-squared error loss that guides the network towards learning a compressed version of the inputs. The action network is itself is trained classically as well using cross-entropy loss, but in addition to backprop affecting it, we backpropagate all the way back into the encoder part of the autoencoder as well. 

### Testing
We intended to test our network by teaching it a widely different set of datasets. MNIST, Bananas Pictures and Car Pictures. We would then make sure that plasticity worked by verifying that the network learned all datasets adequalty without forgetting previosuly learned datasets. Then we would like to show that it encoded "meaning" properly by showing it a [banana-car](https://i.ytimg.com/vi/_9Nm_aI_7hc/maxresdefault.jpg) and showing that it recognizes the presence of a banana as well as a car. 

## Bigger Picture
My research interest is to model the functions of the brain by integrating existing machine learning models together.
Model the entire process by which humans have come to exist using various types of neural networks, and a computational base.

  Energy + Luck + Matter -> Simple cells (+ Time + Natural Selection) -> Humans
  Energy + Human + Matter -> Simple Computation/NN (+ Genetic Algorithm + Time) -> Candidate Nets

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
  - Dynamically Expandable Network (Plasticity)
  - Reinforcement Learning
2. Will learn from it's parent (previous generation of genetic algorithm) and the world,
    much like humans learn from their parents and the world.
3. Will have simple computational bases (genes = code, simple computation = cells that react)
4. It is possible that early generations be "simpler" nets the same way animals appear "simpler" than humans.


#### Step 1:
    Candidate Nets:
        In order to run our overall algorithm we first need a suitable candidate network model.
        We propose a network that integrates the already known neural networks listed above.
            a. First integrate unsupervised learning and plasticity
            b. Evaluate for capability of holding multiple functions
            c. Integrate (a) with recurrent LSTM
            d. Evaluate (b) + capability of information/pattern recognition and memory
            e. Add ability to train on stream of information.
            f. Repeat (d)
        At every evaluation, evaluate this net against known nets.

#### Step 2:
    Natural Selection:
        a. We propose that a genetic algorithm be used to generate and prune generations of candidate nets.
            This algorithm will mainly touch hyper parameters (ie. the computational base).
        b. With the successful achievement of natural selection, we now suggest each generation
            be able to learn from it's parent candidate network by limiting the scope of the genetic algorithm
            and generating a generation of nets based off the best K nets from the previous generation.
            We suggest K=2 from biological inspiration. However this parameter could also be subject to the
            genetic algorithm.

*This repository implements Step 1.a and Step 1.b*
## Software 
### How to Run
1. Install the dependencies below.
2. Pick the experiment you'd like to run, find it's corresponding run_experiment.py file.
3. Run prepare_experiment()
4. Optionally, run find_hypers() to get a good set of hyperparameters.
5. Run train_model(). Parameters are defined inside the function.


### Dependencies
- Python 3.7
- numpy 1.18.4
- Pillow 7.0.0 or Pillow-SIMD 7.0.0
    - For pillow-simd you will need a SSE4.1 capable CPU and `sudo apt-get install libjpeg-dev zlib1g-dev`
- progress 1.5
- torch 1.5.0
- torchvision 0.6.0
- sklearn 0.0
- matplotlib 3.2.1
- Ray[Tune] 0.8.5 (`pip install 'ray[tune]'`)

## Style Guide
We use the industry standard [PEP8].

[PEP8]: <https://pep8.org>

## Resources
Below a compilation of resources we've used along the way, including academic papers, tutorials, etc.
### Papers
- [The wake-sleep algorithm for unsupervised neural networks](https://www.cs.toronto.edu/~hinton/absps/ws.pdf)
- [Lifelong Learning with Dynamically Expandable Networks](https://openreview.net/pdf?id=Sk7KsfW0-) with [code](https://github.com/jaehong-yoon93/DEN)
- [Measuring abstract reasoning in neural networks](https://arxiv.org/pdf/1807.04225.pdf)
### Datasets
- [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- [Car 1](https://archive.ics.uci.edu/ml/datasets/Automobile)
- [Stanford Car](https://www.kaggle.com/jessicali9530/stanford-cars-dataset)
- [Fruit 360](https://www.kaggle.com/moltean/fruits) (organized by subfolders)
- [Mendeley Banana](https://data.mendeley.com/datasets/zk3tkxndjw/2)

## Thanks
Thanks to Prof. John Vervaeke for guidance, thoughts and wisdom.

Thanks to Prof. Steve Mann for providing some computational resources.

Thanks to bjsowa/DEN for inital fork structure.

Thanks to mikacho for guidance on the resource constraining function.
