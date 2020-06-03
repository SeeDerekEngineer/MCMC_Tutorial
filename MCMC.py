import pymc3 as pm
import numpy as np


class MCMC():

    def __init__(self):
        self.myVariable = "Athena"
        self.tNetwork = np.array([[0.9,0.075,0.025], [0.15,0.8,0.05], [0.25,0.25,0.5]])
        self.initState = np.array([0, 1, 0])

    def getMCDistribution(self):
        epsilon = 1
        while epsilon > 0.001:
            nextState = np.dot(self.initState, self.tNetwork)
            epsilon = np.sqrt(np.sum(np.square(nextState - self.initState)))
            self.initState = nextState
        return nextState


