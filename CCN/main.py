# test out network

from cclayer import cclayer
from ccnet import ccnet
import numpy as np

if __name__ == '__main__':

    N = 2
    M = 1
    x = [np.random.rand(N,1) / 2 for _ in range(100)]
    for i in range(50, 100):
        x[i] *= 2
        
    targets = [np.zeros((M, 1)) for _ in range(100)]
    targets[50:] = [np.ones((M, 1)) for _ in range(50)]
    
    print(len(x))
    print(len(targets))

    network = ccnet(N, M, 0.01, 0.5)
    network.create_candidate_layer()
    network.create_candidate_layer()
    network.create_candidate_layer()
    
    epochs = 10
    network.train_candidate_network(x, targets, epochs)
