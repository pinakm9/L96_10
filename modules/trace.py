import numpy as np
import scipy 
import pandas as pd 
import matplotlib.pyplot as plt
import tables
import utility as ut

class Trace:
    
    def __init__(self, files, obs_gap) -> None:
        self.files = files 
        self.obs_gap = obs_gap 

    @ut.timer
    def read(self, file, k=None):
        h5 = tables.open_file(file, mode='r')
        T = len(h5.root.observation.read().tolist()) 
        T = T if k is None else min(k, T)
        trace = np.zeros(T)
        for i in range(T):
            particles = np.array(getattr(h5.root.particles, 'time_' + str(i)).read().tolist())
            evals, _ = np.linalg.eigh(np.cov(particles.T))
            trace[i] = evals.sum()

        phy_time = self.obs_gap * np.arange(0., len(trace), 1.)
        h5.close()
        return trace, phy_time 


    def collect_data(self, k=None, tail=0.7):
        trace = []
        phy_time = []
        self.tail_mean = 0.

        for i, file in enumerate(self.files):
            d, t = self.read(file, k)
            trace.append(d)
            phy_time.append(t)
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        return np.array(trace).reshape(-1), np.array(phy_time).reshape(-1) 


    def collect_mean_data(self, k=None, tail=0.7):
        self.tail_mean = 0.
      
        for i, file in enumerate(self.files):
            d, t =  self.read(file, k)
            if i == 0:
                trace, phy_time = d, t
            else:
                trace += d
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        self.trace, self.phy_time =  trace / len(self.files), phy_time / len(self.files)
    

    def fit_exp(self, tail=0.7):
        self.collect_mean_data(tail=tail)
        def func(x, a, b, c):
            return a * np.exp(b*x) + c#self.tail_mean
        self.popt, self.pcov = scipy.optimize.curve_fit(func, self.phy_time, self.trace, p0=[1.0, -1.0, self.tail_mean]) 
        ss_res = ((self.trace - func(self.phy_time, *self.popt))**2).sum()
        ss_tot = ((self.trace - np.mean(self.trace))**2).sum()
        self.r_squared = 1 - (ss_res / ss_tot)


    def f(self, x):
        a, b, c = self.popt
        return a * np.exp(b*x) + c#self.tail_mean



    #"""
    def plot_line(self, folder, tag, obs_cov):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        trace, phy_time = self.collect_data()
        ax.scatter(phy_time, trace, s=3, c='grey', alpha=0.3)
        ax.plot(self.phy_time, self.trace, c='black', label=r'mean trace', linestyle='dashed')
        ax.set_ylabel(r'posterior trace', fontsize=20)
        ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
        ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}'.format(self.obs_gap, obs_cov), fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('{}/trace_{}.png'.format(folder, tag))
    #"""


class BatchTr:
    
    def __init__(self, file_dicts):
        self.file_dicts = file_dicts 

    def plot(self, folder, tag, obs_cov, ylim, fsize=30, labels=[], linestyles=[]):
        fig = plt.figure(figsize=(8 * len(self.file_dicts[0]), 8))
        axs = []
        for i, obs_gap in enumerate(self.file_dicts[0]):
            if i == 0:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1))
                axs[i].set_ylabel(r'mean posterior trace', fontsize=fsize)
            else:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1, sharey=axs[0], sharex=axs[0]))
                axs[i].get_yaxis().set_visible(False)

            axs[i].tick_params(axis='both', which='major', labelsize=fsize)
            axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
            axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(obs_gap, obs_cov), fontsize=fsize)
            axs[i].set_xlabel(r'time ($t=ng$)', fontsize=fsize)
            if ylim is not None:
                axs[i].set_ylim(*ylim)

            for j, file_dict in enumerate(self.file_dicts):
                trace = Trace(file_dict[obs_gap], obs_gap)
                trace.collect_mean_data()
                axs[i].plot(trace.phy_time, trace.trace, c='black', label=labels[j], linestyle=linestyles[j])
                
            
            axs[i].legend(fontsize=fsize, loc='upper right')
            
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/trace_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)