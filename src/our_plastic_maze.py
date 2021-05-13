# Backpropamine: differentiable neuromdulated plasticity.
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License file in this repository for the specific language governing
# permissions and limitations under the License.

# This code implements the "Grid Maze" task. See section 4.2 in
# Miconi et al. ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym )
# or section 4.5 in Miconi et al.
# ICML 2018 ( https://arxiv.org/abs/1804.02464 )


# The Network class implements a "backpropamine" network, that is, a neural
# network with neuromodulated Hebbian plastic connections that is trained by
# gradient descent. The Backpropamine machinery is
# entirely contained in the Network class (~25 lines of code).

# The rest of the code implements a simple
# A2C algorithm to train the network for the Grid Maze task.


import argparse
import pdb
# from line_profiler import LineProfiler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform
from src.models.PlasticLinear import PlasticRNNMaze as Network

import numpy as np

np.set_printoptions(precision=4)

ADDITIONALINPUTS = 4  # 1 input for the previous reward, 1 input for numstep, 1 unused,  1 "Bias" input

NBACTIONS = 4  # U, D, L, R

RFSIZE = 3  # Receptive Field: RFSIZE x RFSIZE

TOTALNBINPUTS = RFSIZE * RFSIZE + ADDITIONALINPUTS + NBACTIONS

def train(paramdict):
    # params = dict(click.get_current_context().params)

    # TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDITIONALINPUTS + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    # params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    # params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "btchFixmod_" + "".join([str(x) + "_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[
        0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in
                                      sorted(zip(params.keys(), params.values()), key=lambda x: x[0]) for x in pair])[
                             :-1] + "_rngseed_" + str(
        params['rngseed'])  # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']);
    random.seed(params['rngseed']);
    torch.manual_seed(params['rngseed'])

    print("Initializing network")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Network(TOTALNBINPUTS, params['hs']).to(device)  # Creating the network

    print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print("Size (numel) of all optimized elements:", allsizes)
    print("Total size (numel) of all optimized elements:", sum(allsizes))

    # total_loss = 0.0
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'], eps=1e-4, weight_decay=params['l2'])
    # optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

    BATCHSIZE = params['bs']

    LABSIZE = params['msize']
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2

    # Grid maze
    lab[1:LABSIZE - 1, 1:LABSIZE - 1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR, CTR] = 0

    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))

    pos = 0
    # hidden = net.initialZeroState(BATCHSIZE)
    # hebb = net.initialZeroHebb(BATCHSIZE)
    # pw = net.initialZeroPlasticWeights()  # For eligibility traces

    # celoss = torch.nn.CrossEntropyLoss() # For supervised learning - not used here

    print("Starting episodes!")

    for numiter in range(params['nbiter']):

        PRINTTRACE = 0
        if (numiter + 1) % (params['pe']) == 0:
            PRINTTRACE = 1

        # lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        # count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center
        posr = {};
        posc = {}
        rposr = {};
        rposc = {}
        for nb in range(BATCHSIZE):
            # Note: it doesn't matter if the reward is on the center (see below). All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0;
            myrposc = 0
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr;
            rposc[nb] = myrposc
            # print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR

        optimizer.zero_grad()
        loss = 0
        lossv = 0
        # hidden = net.initialZeroState(BATCHSIZE).to(device)
        # hebb = net.initialZeroHebb(BATCHSIZE).to(device)
        numactionchosen = 0

        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        # vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        # reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

        # print("EPISODE ", numiter)
        for numstep in range(params['eplen']):

            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32')

            labg = lab.copy()
            for nb in range(BATCHSIZE):
                inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE // 2:posr[nb] + RFSIZE // 2 + 1,
                                                posc[nb] - RFSIZE // 2:posc[nb] + RFSIZE // 2 + 1].flatten() * 1.0

                # Previous chosen action
                inputs[nb, RFSIZE * RFSIZE + 1] = 1.0  # Bias neuron
                inputs[nb, RFSIZE * RFSIZE + 2] = numstep / params['eplen']
                inputs[nb, RFSIZE * RFSIZE + 3] = 1.0 * reward[nb]
                inputs[nb, RFSIZE * RFSIZE + ADDITIONALINPUTS + numactionschosen[nb]] = 1

            inputsC = torch.from_numpy(inputs).to(device)

            ## Running the network
            y  = net(inputsC)  # y  should output raw scores, not probas

            y = torch.softmax(y, dim=1)
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()
            logprobs.append(distrib.log_prob(actionschosen))
            numactionschosen = actionschosen.data.cpu().numpy()  # We want to break gradients
            reward = np.zeros(BATCHSIZE, dtype='float32')

            for nb in range(BATCHSIZE):
                myreward = 0
                numactionchosen = numactionschosen[nb]

                tgtposc = posc[nb]
                tgtposr = posr[nb]
                if numactionchosen == 0:  # Up
                    tgtposr -= 1
                elif numactionchosen == 1:  # Down
                    tgtposr += 1
                elif numactionchosen == 2:  # Left
                    tgtposc -= 1
                elif numactionchosen == 3:  # Right
                    tgtposc += 1
                else:
                    raise ValueError("Wrong Action")

                reward[nb] = 0.0  # The reward for this step
                if lab[tgtposr][tgtposc] == 1:
                    reward[nb] -= params['wp']
                else:
                    posc[nb] = tgtposc
                    posr[nb] = tgtposr

                # Did we hit the reward location ? Increase reward and teleport!
                # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move...
                # But we still avoid it.
                if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                    reward[nb] += params['rew']
                    posr[nb] = np.random.randint(1, LABSIZE - 1)
                    posc[nb] = np.random.randint(1, LABSIZE - 1)
                    while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                        posr[nb] = np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)

            rewards.append(reward)
            # vs.append(v)
            sumreward += reward

            # This is an "entropy penalty", implemented by the sum-of-squares of the probabilities because our version of PyTorch did not have an entropy() function.
            # The result is the same: to penalize concentration, i.e. encourage diversity in chosen actions.
            loss += (params['bent'] * y.pow(2).sum() / BATCHSIZE)

            # if PRINTTRACE:
            #    print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
            #            #" - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())),
            #            " -Reward (this step, 1st in batch): ", reward[0])

        # Episode is done, now let's do the actual computations of rewards and losses for the A2C algorithm

        # R = torch.zeros(BATCHSIZE).to(device)
        # gammaR = params['gr']
        # for numstepb in reversed(range(params['eplen'])):
        #     R = gammaR * R + torch.from_numpy(rewards[numstepb]).to(device)
        #     ctrR = R - vs[numstepb][0]
        #     lossv += ctrR.pow(2).sum() / BATCHSIZE
        #     loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE
        #     # pdb.set_trace()
        #
        # loss += params['blossv'] * lossv
        # loss /= params['eplen']

        if PRINTTRACE:
            if True:  # params['algo'] == 'A3C':
                print("lossv: ", float(lossv))
            print("Total reward for this episode (all):", sumreward, "Dist:", dist)

        loss.backward()
        all_grad_norms.append(torch.nn.utils.clip_grad_norm(net.parameters(), params['gc']))
        if numiter > 100:  # Burn-in period for meanrewards
            optimizer.step()

        lossnum = float(loss)
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())
        # all_losses_v.append(lossv.data[0])
        # total_loss  += lossnum

        if (numiter + 1) % params['pe'] == 0:
            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ",
                  np.sum(all_total_rewards[-params['pe']:]) / params['pe'])
            # print("Mean reward (across batch): ", sumreward.mean())
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            # print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())

        if (numiter + 1) % params['save_every'] == 0:
            print("Saving files...")
            losslast100 = np.mean(all_losses_objective[-100:])
            print("Average loss over the last 100 episodes:", losslast100)
            print("Saving local files...")
            with open('grad_' + suffix + '.txt', 'w') as thefile:
                for item in all_grad_norms[::10]:
                    thefile.write("%s\n" % item)
            with open('loss_' + suffix + '.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                    thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_' + suffix + '.dat')
            with open('params_' + suffix + '.dat', 'wb') as fo:
                pickle.dump(params, fo)
            if os.path.isdir('/mnt/share/tmiconi'):
                print("Transferring to NFS storage...")
                for fn in ['params_' + suffix + '.dat', 'loss_' + suffix + '.txt', 'torchmodel_' + suffix + '.dat']:
                    result = os.system(
                        'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/' + fn))
                print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value (reward increment for taking correct action after correct stimulus)",
                        default=10.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.0)
    parser.add_argument("--bent", type=float,
                        help="coefficient for the entropy reward (really Simpson index concentration measure)",
                        default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=11)
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float, help="gradient norm clipping", default=4.0)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=200)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=30)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=0)  # 3e-6
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=50)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information",
                        default=10)
    args = parser.parse_args();
    argvars = vars(args);
    argdict = {k: argvars[k] for k in argvars if argvars[k] != None}

    train(argdict)
    # lp = LineProfiler()
    # lpwrapper = lp(train)
    # lpwrapper(argdict)
    # lp.print_stats()

