import numpy as np
import scipy 
import pandas as pd 
import matplotlib.pyplot as plt
import tables

class L2Error:
    
    def __init__(self, files, obs_gap) -> None:
        self.files = files 
        self.obs_gap = obs_gap
        h5 = tables.open_file(files[0], mode='r')
        self.dim = np.array(h5.root.particles.time_0.read().tolist()).shape[-1]
        h5.close()


    def read(self, file, k=None):
        h5 = tables.open_file(file, mode='r')
        l2_error = h5.root.l2_error.read()
        if k is not None:
            l2_error = l2_error[:k]
        phy_time = self.obs_gap * np.arange(0., len(l2_error), 1.)
        h5.close()
        return l2_error / np.sqrt(self.dim), phy_time 


    def collect_data(self, k=None, tail=0.7):
        l2_error = []
        phy_time = []
        self.tail_mean = 0.

        for i, file in enumerate(self.files):
            d, t = self.read(file, k)
            l2_error.append(d)
            phy_time.append(t)
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        return np.array(l2_error).reshape(-1), np.array(phy_time).reshape(-1) 


    def collect_mean_data(self, k=None, tail=0.7):
        self.tail_mean = 0.
      
        for i, file in enumerate(self.files):
            d, t =  self.read(file, k)
            if i == 0:
                l2_error, phy_time = d, t
            else:
                l2_error += d
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.files)
        self.l2_error, self.phy_time = l2_error / len(self.files), phy_time / len(self.files)
    

    def fit_exp(self, tail=0.7):
        self.collect_mean_data(tail=tail)
        def func(x, a, b, c):
            return a * np.exp(b*x) + c#self.tail_mean
        self.popt, self.pcov = scipy.optimize.curve_fit(func, self.phy_time, self.l2_error, p0=[1.0, -1.0, self.tail_mean]) 
        ss_res = ((self.l2_error - func(self.phy_time, *self.popt))**2).sum()
        ss_tot = ((self.l2_error - np.mean(self.l2_error))**2).sum()
        self.r_squared = 1 - (ss_res / ss_tot)


    def f(self, x):
        a, b, c = self.popt
        return a * np.exp(b*x) + c#self.tail_mean



    #"""
    def plot_line(self, folder, tag, obs_cov, save=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        l2_error, phy_time = self.collect_data()
        ax.scatter(phy_time, l2_error, s=3, c='grey', alpha=0.3)
        a, b, c = self.popt
        label = r'${:.2f}\,\exp({:.2f}t) + {:.2f}$'.format(a, b, c)#self.tail_mean)
        #ax.plot(self.phy_time, self.f(self.phy_time), c='deeppink', label=label)
        ax.plot(self.phy_time, np.ones_like(self.phy_time) * np.sqrt(obs_cov), label='$\sigma$')
        ax.plot(self.phy_time, self.l2_error, c='black', label=r'mean error', linestyle='dashed')
        ax.set_ylabel(r'$\frac{\rm error}{\sqrt{\rm dimension}}$', fontsize=20)
        ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
        ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}, $R^2$ = {:.2f}'.format(self.obs_gap, obs_cov, self.r_squared), fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if save:
            plt.savefig('{}/l2_error_{}.png'.format(folder, tag))
        return fig, ax
    #"""
