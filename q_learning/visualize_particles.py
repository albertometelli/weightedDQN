import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os 
import argparse
parser = argparse.ArgumentParser()
arg_graphs = parser.add_argument_group('Graphs')
arg_graphs.add_argument("--particles", action='store_true')
arg_graphs.add_argument("--variance", action='store_true')
arg_graphs.add_argument("--probabilities", action='store_true')
args = parser.parse_args()
from pylab import *

if not args.particles and not args.variance and not args.probabilities:
    raise SystemExit
def _compute_prob_max(q_list):
        q_array = np.array(q_list)
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        prob = prob.astype(np.float32)
        return prob / np.sum(prob)
        
environments = ['RiverSwim']
algorithms = ['particle-ql']
policies = [['weighted', 'vpi']]
updates = [['weighted','mean' ]]
inits=['eq-spaced','q-max','borders']
conf = 0.95
timesteps_to_show=100000
i=0
for env in environments:
    for j, alg in enumerate(algorithms):
        particle_out_dir=env+"/"+alg+"/figures/particles"
        std_out_dir=env+"/"+alg+"/figures/std"
        prob_out_dir=env+"/"+alg+"/figures/prob"
        if not os.path.exists(particle_out_dir):
                os.makedirs(particle_out_dir)
        if not os.path.exists(std_out_dir):
                os.makedirs(std_out_dir)
        if not os.path.exists(prob_out_dir):
                os.makedirs(prob_out_dir)
        for pi in policies[j]:
            for u in updates[j]:
                for init in inits:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    name=pi+"-"+u+"-"+init
                    fig.suptitle(name)
                 
                    paths = glob.glob( alg + "/qs_" + pi + "_*_"+ u + "_*_"+"*_"+init+"*.npy")
                    #print(paths)
                    for p in paths:
                        
                        history = np.load(p)
                        #print(history.shape)
                        timesteps = history.shape[0]
                        n_states=history.shape[2]
                        n_actions=history.shape[3]
                        #show particles
                        if args.particles:
                            ys_1=history[0:timesteps_to_show,:,0,0]
                            ys_2=history[0:timesteps_to_show,:,0,1]
                            #print(ys_1.shape)
                            for k in range(ys_1.shape[1]):
                                y=ys_1[:,k]
                                ax[0].plot(range(len(y)), y)
                            ax[0].set_xlabel("iterations")
                            ax[0].set_ylabel("particles")
                            for k in range(ys_2.shape[1]):
                                y=ys_2[:,k]
                                ax[1].plot(range(len(y)), y)
                            ax[1].set_xlabel("iterations")
                            ax[1].set_ylabel("particles")
                            fig.savefig(particle_out_dir+'/{}.pdf'.format(name),  format='pdf')
                            
                        if args.variance:
                            ys_1=history[0:timesteps_to_show,:,0,0]
                            ys_2=history[0:timesteps_to_show,:,0,1]
                            #show standard deviation of q estimates
                            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                            fig.suptitle(name)
                            stds_1=np.std(ys_1, axis=1)
                            stds_2=np.std(ys_2, axis=1)
                            ax[0].plot(range(len(stds_1)), stds_1)
                            ax[0].set_xlabel("iterations")
                            ax[0].set_ylabel("std of q values")
                            ax[1].plot(range(len(stds_2)), stds_2)
                            ax[1].set_xlabel("iterations")
                            ax[1].set_ylabel("std of q values")
                            ax[0].set_ylim(ymin=0, ymax=np.max(stds_1)+5)
                            ax[1].set_ylim(ymin=0, ymax=np.max(stds_2)+5)
                            fig.savefig(std_out_dir+'/{}.pdf'.format(name), format='pdf')
                            
                        if args.probabilities:
                            #show probabilities
                            history=np.transpose(history, [0, 2, 3, 1])
                            probabilities=np.zeros(shape=(timesteps, n_states))
                        
                        
                            fig.suptitle(name)
                            state=0
                            col=0
                            for s in range(1):
                                for t in range(timesteps):
                                    particles=history[t, s, :, :]
                                    means=np.mean(particles, axis=1)
                                    greedy_actions=np.argwhere(means == np.max(means)).ravel()
                                    probs=_compute_prob_max(particles)
                                    prob = 1-max(probs) 
                                    probabilities[t, s]=prob
                                probs=probabilities[:, s]
                                ax1 = subplot(1, 1,s+1)
                                ax1.plot(range(len(probs)), probs)
                                ax1.set_ylim(ymin=0)
                            plt.suptitle(name)
                            plt.savefig(prob_out_dir+'/{}.pdf'.format(name), format='pdf')
                            print(name+": Done")
                    i+=1
