import numpy as np
from mushroom.environments.finite_mdp import FiniteMDP

def generate_loop(gamma=.99, horizon=np.inf):
        p =compute_probabilities() 
        r=compute_rewards()
        mu=compute_mu()
        return FiniteMDP(p, r, mu, gamma, horizon)

        

         
    
def compute_probabilities():
        p=np.zeros((9, 2, 9))
        p[0, 0, 1]=p[0, 1, 5]=1
        p[1, 0, 2]=p[1, 1, 2]=1
        p[2, 0, 3]=p[2, 1, 3]=1
        p[3, 0,4]=p[3,  1, 4]=1
        p[4, 0, 0]=p[4, 1, 0]=1
        p[5, 0, 0]=p[5, 1, 6]=1
        p[6, 0, 0]=p[6, 1, 7]=1
        p[7, 0, 0]=p[7, 1, 8]=1
        p[8, 0, 0]=p[8, 1, 0]=1
        return p

def compute_rewards():
        r = np.zeros((9, 2, 9))
        r[4, 0, 0]=r[4, 1, 0]=0
        r[8, 0, 0]=r[8, 1, 0]=2
        return r

def compute_mu():
        mu = np.zeros(9)
        mu[0]=1
        return mu
