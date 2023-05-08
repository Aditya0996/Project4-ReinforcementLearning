import matplotlib.pyplot as plt
import os
import numpy as np


def plotQlearning(worldId, epoch, cumulativeAverage, currentRun):
    plt.figure(2)
    plt.plot(cumulativeAverage)
    plt.xscale('log')
    if not os.path.exists(f'runs/world_{worldId}/attempt_{currentRun}'):
        os.makedirs(f'runs/world_{worldId}/attempt_{currentRun}')
    plt.savefig(f'runs/world_{worldId}/attempt_{currentRun}/world_{worldId}_epoch{epoch}learning.png')


def epsilonDecayFunction(epsilon, epoch):
    epsilon = epsilon * np.exp(-.01 * epoch)
    return epsilon
