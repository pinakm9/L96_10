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
        h5 = tables.open_file(files[0], mode='r')
        self.dim = np.array(h5.root.particles.time_0.read().tolist()).shape[-1]
        print('System dimension found to be {}'.format(self.dim))
        h5.close() 

    #@ut.timer
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
        root_trace = []
        phy_time = []
        self.tail_mean = 0.

        for i, file in enumerate(self.files):
            d, t = self.read(file, k)
            root_trace.append(np.sqrt(d))
            phy_time.append(t)
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        return np.array(root_trace).reshape(-1) / np.sqrt(self.dim), np.array(phy_time).reshape(-1) 


    def collect_mean_data(self, k=None, tail=0.7):
        self.tail_mean = 0.
        sqrt_dim = np.sqrt(self.dim)

        for i, file in enumerate(self.files):
            d, t =  self.read(file, k)
            if i == 0:
                root_trace, phy_time = np.sqrt(d)/sqrt_dim, t
            else:
                root_trace += np.sqrt(d)/sqrt_dim
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(np.sqrt(d[t:]) / sqrt_dim)
        self.tail_mean /= len(self.files)
        self.root_trace, self.phy_time =  root_trace / len(self.files), phy_time / len(self.files)
    

    def collect_squared_mean_data(self, k=None, tail=0.7):
        self.tail_mean = 0.

        for i, file in enumerate(self.files):
            d, t =  self.read(file, k)
            if i == 0:
                trace, phy_time = d/self.dim, t
            else:
                trace += d/self.dim
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(np.sqrt(d[t:]) / self.dim)
        self.tail_mean /= len(self.files)
        self.trace, self.phy_time =  trace / len(self.files), phy_time / len(self.files)

    
    
    def fit_exp(self, tail=0.7):
        self.collect_mean_data(tail=tail)
        def func(x, a, b, c):
            return a * np.exp(b*x) + c#self.tail_mean
        self.popt, self.pcov = scipy.optimize.curve_fit(func, self.phy_time, self.root_trace, p0=[1.0, -1.0, self.tail_mean]) 
        ss_res = ((self.root_trace - func(self.phy_time, *self.popt))**2).sum()
        ss_tot = ((self.root_trace - np.mean(self.root_trace))**2).sum()
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
        ax.plot(self.phy_time, self.root_trace, c='black', label=r'mean trace', linestyle='dashed')
        ax.set_ylabel(r'posterior trace', fontsize=20)
        ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
        ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}'.format(self.obs_gap, obs_cov), fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('{}/root_trace_{}.png'.format(folder, tag))
    #"""


class BatchTr:
    
    def __init__(self, file_dicts):
        self.file_dicts = file_dicts 

    def plot(self, folder, tag, obs_cov, ylim, fsize=30, labels=[], linestyles=[], linewidth=5.):
        fig = plt.figure(figsize=(8 * len(self.file_dicts[0]), 8))
        axs = [] 
        for i, obs_gap in enumerate(self.file_dicts[0]):
            if i == 0:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1))
                axs[i].set_ylabel(r'$\mathbb{E}[s_n^2]$', fontsize=fsize+10)
                axs[i].set_xlabel(r'time ($t=ng$)', fontsize=fsize+10)
            else:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1, sharey=axs[0], sharex=axs[0]))
                axs[i].get_yaxis().set_visible(False)

            axs[i].tick_params(axis='both', which='major', labelsize=fsize)
            axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
            #axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(obs_gap, obs_cov), fontsize=fsize)
            
            if ylim is not None:
                axs[i].set_ylim(*ylim)

            for j, file_dict in enumerate(self.file_dicts):
                trace = Trace(file_dict[obs_gap], obs_gap)
                trace.collect_squared_mean_data()
                axs[i].plot(trace.phy_time, trace.trace, c='black', label=labels[j], linestyle=linestyles[j], linewidth=linewidth)
                
            axs[i].plot(trace.phy_time, np.ones_like(trace.phy_time) * obs_cov, label='$\sigma^2$', linestyle='dotted', c='black', linewidth=linewidth)
            if i == len(self.file_dicts[0]) - 1:
                axs[i].legend(fontsize=fsize, loc='upper right')
            
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/trace_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)


class BatchTr2:
    
    def __init__(self, file_dicts, obs_gap):
        self.file_dicts = file_dicts 
        self.obs_gap = obs_gap

    def plot(self, folder, tag, ylim, fsize=30, labels=[], linestyles=[], linewidth=5.):
        fig = plt.figure(figsize=(8 * len(self.file_dicts[0]), 8))
        axs = []
        for i, obs_cov in enumerate(self.file_dicts[0]):
            if i == 0:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1))
                axs[i].set_ylabel(r'$\mathbb{E}[s_n^2]$', fontsize=fsize+10)
                axs[i].set_xlabel(r'time ($t=ng$)', fontsize=fsize+10)
            else:
                axs.append(fig.add_subplot(1, len(self.file_dicts[0]), i+1, sharey=axs[0], sharex=axs[0]))
                axs[i].get_yaxis().set_visible(False)

            axs[i].tick_params(axis='both', which='major', labelsize=fsize)
            axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
            #axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(self.obs_gap, obs_cov), fontsize=fsize)
            
            if ylim is not None:
                axs[i].set_ylim(*ylim)

            for j, file_dict in enumerate(self.file_dicts):
                trace = Trace(file_dict[obs_cov], self.obs_gap)
                trace.collect_squared_mean_data()
                print("max trace = {:.4f},{:.4f}".format(max(trace.trace), max(np.sqrt(trace.trace))))
                axs[i].plot(trace.phy_time, trace.trace, c='black', label=labels[j], linestyle=linestyles[j], linewidth=linewidth)
            axs[i].plot(trace.phy_time, np.ones_like(trace.phy_time) * obs_cov, label='$\sigma^2$', linestyle='dotted', c='black', linewidth=linewidth)    
            if i == len(self.file_dicts[0]) - 1:
                axs[i].legend(fontsize=fsize, loc='upper right')
            
            
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/trace_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)
        