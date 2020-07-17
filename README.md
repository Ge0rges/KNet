<div align="center">

# The Plastic Support Structure

**A network augmentation that allows for efficient multi-task training.**

[![CodeFactor](https://www.codefactor.io/repository/github/ge0rges/knet/badge/master?s=2bd0ecc26d1a05b4bb8b1a7855a145c0cce0baf9)](https://www.codefactor.io/repository/github/ge0rges/knet/overview/master)

</div>

## Detailed High Level Process
When a neuron has high semantic drift after training a new task, duplicate it and revert it to it's old weight value. 
For each layer in which a neuron has not been split, add a new neuron.
Connect all new neurons together, and in the last hidden layer to the appropriate output neuron.
Set the outgoing weights of new neurons to old neurons to zero. Train these new neurons exclusively.

## Software 
# How to Run
1. Install the dependencies below.
2. Pick the experiment you'd like to run, find it's corresponding run_experiment.py file.
3. Run prepare_experiment()
4. Optionally, run find_hypers() to get a good set of hyperparameters.
5. Run train_model(). Parameters are defined inside the function.


# Dependencies
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
- [Lifelong Learning with Dynamically Expandable Networks](https://openreview.net/pdf?id=Sk7KsfW0-) with [code](https://github.com/jaehong-yoon93/DEN)
### Datasets
- MNIST
- MNIST Variations

## Thanks
Thanks to Prof. John Vervaeke for guidance, thoughts and wisdom.

Thanks to Prof. Steve Mann for providing some computational resources.

Thanks to bjsowa/DEN for inital fork structure.

Thanks to mikacho for guidance on the resource constraining function.


## Citation
Unavailable
