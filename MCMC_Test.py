import unittest
import MCMC as mcmc
import numpy as np


class MCMC_Test(unittest.TestCase):

    def setUp(self):
        self.myMCMC = mcmc.MCMC()

    def testCreation(self):
        self.assertEqual(self.myMCMC.tNetwork.tolist(),
                         np.array([[0.9,0.075,0.025], [0.15,0.8,0.05], [0.25,0.25,0.5]]).tolist())

    def testNextState(self):
        self.assertEqual(self.myMCMC.getMCDistribution().tolist(),
                         np.array([0.6233219338533376, 0.3140341304711523, 0.06264393567551067]).tolist())

if __name__ == "__main__":
    unittest.main()