from email.header import Header
import numpy as np
import scipy 
import pandas as pd 
import matplotlib.pyplot as plt
import tables
import utility as ut
from scipy.stats import pearsonr

class Dvs:
    
    def __init__(self, dist_files, trace_files, obs_gap) -> None:
        self.d_files = dist_files 
        self.t_files = trace_files
        self.obs_gap = obs_gap
        h5 = tables.open_file(trace_files[0], mode='r')
        self.dim = np.array(h5.root.particles.time_0.read().tolist()).shape[-1]
        h5.close()


    #@ut.timer
    def read_trace_file(self, file, k=None):
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

        

    def read_d(self, file, k=None):
        dist = pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()
        if k is not None:
            dist = dist[:k]
        phy_time = self.obs_gap * np.arange(0., len(dist), 1.)
        return dist, phy_time  


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


    def collect_mean_data_s(self, k=None, tail=0.7):
        self.tail_mean = 0.
        for i, file in enumerate(self.t_files):
            d, t =  self.read_trace_file(file, k)
            if i == 0:
                root_trace, phy_time = np.sqrt(d/self.dim), t
            else:
                root_trace += np.sqrt(d/self.dim)
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(np.sqrt(d[t:]) / self.dim)
        self.tail_mean /= len(self.t_files)
        self.root_trace, self.phy_time =  root_trace / len(self.t_files), phy_time / len(self.t_files)


    def collect_mean_data_d(self, k=None, tail=0.7):
        self.tail_mean = 0.
      
        for i, file in enumerate(self.d_files):
            d, t =  self.read_d(file, k)
            if i == 0:
                dist, phy_time = d, t
            else:
                dist += d
                phy_time += t
            t = int(len(d) * tail)
            self.tail_mean += np.mean(d[t:])
        self.tail_mean /= len(self.d_files)
        self.dist, self.phy_time = dist / len(self.d_files), phy_time / len(self.d_files)

    
    

    def fit_line(self, tail=0.7):
        self.collect_mean_data_d(tail=tail)
        self.collect_mean_data_s(tail=tail)
        def func(x, a, b):
            return a*x + b
        self.popt, self.pcov = scipy.optimize.curve_fit(func, self.dist, self.root_trace, p0=[1.0, 0.0]) 
        ss_res = ((self.root_trace - func(self.dist, *self.popt))**2).sum()
        ss_tot = ((self.root_trace - np.mean(self.root_trace))**2).sum()
        self.r_squared = 1 - (ss_res / ss_tot)


    def f(self, x):
        a, b = self.popt
        return a*x + b



    #"""
    def plot_line(self, folder, tag, obs_cov):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
   
        ax.scatter(self.dist, self.trace, c='black', s=10)
        label = r'${:.2f}D_\varepsilon {} {:.2f}, R^2 = {:.2f}$'.format(self.popt[0], '' if self.popt[1] < 0 else '+', self.popt[1], self.r_squared)
        ax.plot(self.dist, self.f(self.dist), label=label)
        ax.set_ylabel(r'mean $\frac{\rm error}{\sqrt{\rm dimension}}$', fontsize=20)
        ax.set_xlabel(r'mean $D_\varepsilon$', fontsize=20)
        ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}, Pearson correlation = {:.2f}'\
            .format(self.obs_gap, obs_cov, pearsonr(self.dist, self.trace)[0]), fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('{}/d_vs_l2_{}.png'.format(folder, tag))
    #"""


class BatchDvs:
    
    def __init__(self, dist_file_dict, trace_file_dict):
        self.dvss = [Dvs(dist_file_dict[obs_gap], trace_file_dict[obs_gap], obs_gap) for obs_gap in dist_file_dict]
        self.dist_file_dict = dist_file_dict
        self.trace_file_dict = trace_file_dict 

    def plot(self, folder, tag, obs_cov, ylim, xlim, fsize=30, linewidth=5.):
        fig = plt.figure(figsize=(8 * len(self.dist_file_dict), 8))
        axs = []
        with open('{}/p_dvs_{}.txt'.format(folder, tag), 'w') as file:
            for i, obs_gap in enumerate(self.dist_file_dict):
                self.dvss[i].fit_line(tail=0.9)
                file.write('{0:.2e} $\pm$ {2:.2e} & {1:.2e} $\pm$ {3:.2e}\\\\\n\\hline\n'.format(*self.dvss[i].popt, *(1.96 * np.diag(self.dvss[i].pcov))))
                if i == 0:
                    axs.append(fig.add_subplot(1, len(self.dist_file_dict), i+1))
                    axs[i].set_ylabel(r'$\mathbb{E}[s_n(\mu_b)]$', fontsize=fsize+10)
                    axs[i].set_xlabel(r'$\mathbb{E}[D_\varepsilon\left(\pi_n(\mu_0), \pi_n(\mu_b)\right)]$', fontsize=fsize+10)
                else:
                    axs.append(fig.add_subplot(1, len(self.dist_file_dict), i+1, sharey=axs[0], sharex=axs[0]))
                    axs[i].get_yaxis().set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=fsize)
                axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
                
                axs[i].scatter(self.dvss[i].dist, self.dvss[i].root_trace, c='black', s=100)
                label = r'${:.2f}D_\varepsilon {} {:.2f}$'.format(self.dvss[i].popt[0], '' if self.dvss[i].popt[1] < 0 else '+', self.dvss[i].popt[1])
                axs[i].plot(self.dvss[i].dist, self.dvss[i].f(self.dvss[i].dist), c='darkgrey', label=label, linewidth=linewidth)
                
                
                #axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(obs_gap, obs_cov), fontsize=fsize)
                axs[i].text(6.0, 1.0, r'Corr = {:.2f}'.format(pearsonr(self.dvss[i].dist, self.dvss[i].root_trace)[0]), fontsize=fsize)
                axs[i].text(6.0, 1.5, r'$R^2$ = {:.2f}'.format(self.dvss[i].r_squared), fontsize=fsize)
                axs[i].legend(fontsize=fsize-0, loc='upper left')
                if ylim is not None:
                    axs[i].set_ylim(*ylim)
                if xlim is not None:
                    axs[i].set_xlim(*xlim)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/dvs_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)
        file.close()



class BatchDvs2:
    
    def __init__(self, dist_file_dict, trace_file_dict, obs_gap):
        self.dvss = [Dvs(dist_file_dict[obs_cov], trace_file_dict[obs_cov], obs_cov) for obs_cov in dist_file_dict]
        self.dist_file_dict = dist_file_dict
        self.root_trace_file_dict = trace_file_dict
        self.obs_gap = obs_gap

    def plot(self, folder, tag, ylim, xlim, fsize=30, linewidth=5.):
        fig = plt.figure(figsize=(8 * len(self.dist_file_dict), 8))
        axs = []
        with open('{}/p_dvs_{}.txt'.format(folder, tag), 'w') as file:
            for i, obs_cov in enumerate(self.dist_file_dict):
                self.dvss[i].fit_line(tail=0.9)
                file.write('{0:.2e} $\pm$ {2:.2e} & {1:.2e} $\pm$ {3:.2e}\\\\\n\\hline\n'.format(*self.dvss[i].popt, *(1.96 * np.diag(self.dvss[i].pcov))))
                if i == 0:
                    axs.append(fig.add_subplot(1, len(self.dist_file_dict), i+1))
                    axs[i].set_ylabel(r'$\mathbb{E}[s_n(\mu_b)]$', fontsize=fsize+10)
                    axs[i].set_xlabel(r'$\mathbb{E}[D_\varepsilon\left(\pi_n(\mu_0), \pi_n(\mu_b)\right)]$', fontsize=fsize+10)
                else:
                    axs.append(fig.add_subplot(1, len(self.dist_file_dict), i+1, sharey=axs[0], sharex=axs[0]))
                    axs[i].get_yaxis().set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=fsize)
                axs[i].tick_params(axis='both', which='minor', labelsize=fsize)
                
                axs[i].scatter(self.dvss[i].dist, self.dvss[i].root_trace, c='black', s=100)
                label = r'${:.2f}D_\varepsilon {} {:.2f}$'.format(self.dvss[i].popt[0], '' if self.dvss[i].popt[1] < 0 else '+', self.dvss[i].popt[1])
                axs[i].plot(self.dvss[i].dist, self.dvss[i].f(self.dvss[i].dist), c='darkgrey', label=label, linewidth=linewidth)
                
                
               #axs[i].set_title(r'$g = {:.2f},\,\sigma= {:.2f}$'.format(self.obs_gap, obs_cov), fontsize=fsize)
                axs[i].text(6.0, 1.0, r'Corr = {:.2f}'.format(pearsonr(self.dvss[i].dist, self.dvss[i].root_trace)[0]), fontsize=fsize)
                axs[i].text(6.0, 1.5, r'$R^2$ = {:.2f}'.format(self.dvss[i].r_squared), fontsize=fsize)
                axs[i].legend(fontsize=fsize-0, loc='upper left')
                if ylim is not None:
                    axs[i].set_ylim(*ylim)
                if xlim is not None:
                    axs[i].set_xlim(*xlim)
                print("max trace = {:.4f}".format(max(self.dvss[i].root_trace)))
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/dvs_{}.png'.format(folder, tag), dpi=300, bbox_inches='tight', pad_inches=0)
        file.close()
