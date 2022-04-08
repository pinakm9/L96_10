import numpy as np
import scipy 
import pandas as pd 
import matplotlib.pyplot as plt

class RateCalc:
    
    def __init__(self, files, obs_gap) -> None:
        self.files = files 
        self.obs_gap = obs_gap 


    def read(self, file, k=None):
        dist = pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()
        if k is not None:
            dist = dist[:k]
        phy_time = self.obs_gap * np.arange(0., len(dist), 1.)
        return dist, phy_time 


    def collect_data(self, k=None, tail=0.7):
        dist = []
        phy_time = []
        self.tail_mean = 0.

        for i, file in enumerate(self.files):
            d, t = self.read(file, k)
            dist.append(d)
            phy_time.append(t)
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        return np.array(dist).reshape(-1), np.array(phy_time).reshape(-1) 


    def collect_mean_data(self, k=None, tail=0.7):
        self.tail_mean = 0.
      
        for i, file in enumerate(self.files):
            d, t =  self.read(file, k)
            if i == 0:
                dist, phy_time = d, t
            else:
                dist += d
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        self.dist, self.phy_time = dist / len(self.files), phy_time / len(self.files)
    

    def fit_exp(self, tail=0.7):
        self.collect_mean_data(tail=tail)
        def func(x, a, b, c):
            return a * np.exp(b*x) + c#self.tail_mean
        self.popt, self.pcov = scipy.optimize.curve_fit(func, self.phy_time, self.dist, p0=[1.0, -1.0, self.tail_mean]) 
        ss_res = ((self.dist - func(self.phy_time, *self.popt))**2).sum()
        ss_tot = ((self.dist - np.mean(self.dist))**2).sum()
        self.r_squared = 1 - (ss_res / ss_tot)


    def f(self, x):
        a, b, c = self.popt
        return a * np.exp(b*x) + c#self.tail_mean



    #"""
    def plot_line(self, folder, tag, obs_cov, ylim=None):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        dist, phy_time = self.collect_data()
        ax.scatter(phy_time, dist, s=3, c='grey', alpha=0.3)
        label = r'${:.2f}\,\exp({:.2f}t) + {:.2f}, R^2 = {:.2f}$'.format(*self.popt, self.r_squared)
        ax.plot(self.phy_time, self.f(self.phy_time), c='deeppink', label=label)
        ax.plot(self.phy_time, self.dist, c='black', label=r'mean $D_\varepsilon$', linestyle='dashed')
        ax.set_ylabel(r'$D_\varepsilon\left(\pi_n(\mu_1), \pi_n(\mu_2)\right)$', fontsize=20)
        ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
        ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}'.format(self.obs_gap, obs_cov), fontsize=20)
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('{}/rate_{}.png'.format(folder, tag))
    #"""


class BatchRate:
    
    def __init__(self, file_dict):
        self.rcs = [RateCalc(file_dict[obs_gap], obs_gap) for obs_gap in file_dict]
        self.file_dict = file_dict 

    def plot(self, folder, tag, obs_cov, ylim, fsize=30):
        fig = plt.figure(figsize=(8 * len(self.file_dict), 8))
        axs = []
        for i, obs_gap in enumerate(self.file_dict):
            self.rcs[i].fit_exp(tail=0.9)
            if i == 0:
                axs.append(fig.add_subplot(1, len(self.file_dict), i+1))
                axs[i].set_ylabel(r'$D_\varepsilon\left(\pi_n(\mu_1), \pi_n(\mu_2)\right)$', fontsize=fsize)
            else:
                axs.append(fig.add_subplot(1, len(self.file_dict), i+1, sharey=axs[0], sharex=axs[0]))
                axs[i].get_yaxis().set_visible(False)
            axs[i].tick_params(axis='both', which='major', labelsize=fsize)
            axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
            dist, phy_time = self.rcs[i].collect_data()
            axs[i].scatter(phy_time, dist, s=10, c='grey', alpha=0.3)
            label = r'${:.2f}\,\exp({:.2f}t) + {:.2f}$'.format(*self.rcs[i].popt)
            axs[i].plot(self.rcs[i].phy_time, self.rcs[i].f(self.rcs[i].phy_time), c='black', label=label)
            axs[i].plot(self.rcs[i].phy_time, self.rcs[i].dist, c='black', label=r'mean $D_\varepsilon$', linestyle='dashed')
            axs[i].text(0.9, 8.0, r'$R^2$ for fit = {:.2f}'.format(self.rcs[i].r_squared), fontsize=fsize)
            axs[i].set_xlabel(r'time ($t=ng$)', fontsize=fsize)
            axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(obs_gap, obs_cov), fontsize=fsize)
            axs[i].legend(fontsize=fsize-0, loc='upper right')
            if ylim is not None:
                axs[i].set_ylim(*ylim)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/rate_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)




class BatchRate2:
    
    def __init__(self, file_dict, obs_gap):
        self.rcs = [RateCalc(file_dict[obs_cov], obs_gap) for obs_cov in file_dict]
        self.file_dict = file_dict 

    def plot(self, folder, tag, ylim, fsize=30):
        fig = plt.figure(figsize=(8 * len(self.file_dict), 8))
        axs = []
        for i, obs_cov in enumerate(self.file_dict):
            self.rcs[i].fit_exp(tail=0.9)
            if i == 0:
                axs.append(fig.add_subplot(1, len(self.file_dict), i+1))
                axs[i].set_ylabel(r'$D_\varepsilon\left(\pi_n(\mu_1), \pi_n(\mu_2)\right)$', fontsize=fsize)
            else:
                axs.append(fig.add_subplot(1, len(self.file_dict), i+1, sharey=axs[0], sharex=axs[0]))
                axs[i].get_yaxis().set_visible(False)
            axs[i].tick_params(axis='both', which='major', labelsize=fsize)
            axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
            dist, phy_time = self.rcs[i].collect_data()
            axs[i].scatter(phy_time, dist, s=10, c='grey', alpha=0.3)
            label = r'${:.2f}\,\exp({:.2f}t) + {:.2f}$'.format(*self.rcs[i].popt)
            axs[i].plot(self.rcs[i].phy_time, self.rcs[i].f(self.rcs[i].phy_time), c='black', label=label)
            axs[i].plot(self.rcs[i].phy_time, self.rcs[i].dist, c='black', label=r'mean $D_\varepsilon$', linestyle='dashed')
            axs[i].text(0.9, 8.0, r'$R^2$ for fit = {:.2f}'.format(self.rcs[i].r_squared), fontsize=fsize)
            axs[i].set_xlabel(r'time ($t=ng$)', fontsize=fsize)
            axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(self.rcs[i].obs_gap, obs_cov), fontsize=fsize)
            axs[i].legend(fontsize=fsize-0, loc='upper right')
            if ylim is not None:
                axs[i].set_ylim(*ylim)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/rate_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)