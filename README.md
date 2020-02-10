# KNet
## Description
This network aims to better model two of the brain's core processes that are said to be critical to learning and cognition: plasticity and unsupervised learning. The network aims to do this by being a hybrid of Cascade Correlational Model which models plasticity, and an unsupervised neural network based on the wake sleep algorithm.

So far, our design has led us to having a wake-sleep network whose layers are CCNs with a variable number of nodes.

## Detailed High Level Idea
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

## Software 
# Dependencies
- Python 3.7
- Numpy 1.18.1
- Pillow (PIL Fork) 7.0.0
- Pytorch 1.4.0
- TorchVision 0.5.0

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
