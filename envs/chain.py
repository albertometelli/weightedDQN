import numpy as np
from mushroom.environments.finite_mdp import FiniteMDP

def generate_chain(n=5, slip=0.2, small=2, large=10, gamma=0.99, horizon=1000):
        nA = 2
        nS = n
        p = compute_probabilities(slip,nS, nA)
        r = compute_rewards(nS, nA, small, large)
        mu = compute_mu(nS)
        return FiniteMDP(p, r, mu, gamma, horizon)

        

         
    
def compute_probabilities(slip, nS, nA):
        p=np.zeros((nS, nA, nS))
        for i in range(nS):
            p[i, 0, min(nS-1, i+1)] = 1-slip
            for k in range(i+1):
                p[i, 1, k] = (1 - slip) / (i + 1)
            #p[i, 1, max(0, i-1)] = 1-slip
            p[i, 1, 0] += slip
            p[i, 0, 0] = slip
        #p[0, 1, 0] = 1.
        #p[1, 1, 0] = (1. - slip) / 2 + slip
        return p

def compute_rewards(nS, nA, small, large):
        r = np.zeros((nS, nA, nS))
        for i in range(nS):
           r[i, 1, 0]=r[i, 0, 0]=small
        r[nS-1, 0, nS-1]=r[nS-1, 1, nS-1] = large
        return r

def compute_mu(nS):
        mu = np.zeros(nS)
        mu[0]=1
        return mu
