import numpy as np
import scipy 
import pandas as pd 
import matplotlib.pyplot as plt
import tables

class Distribution:
    
    def __init__(self, files, obs_gap) -> None:
        self.files = files 
        self.obs_gap = obs_gap
        h5 = tables.open_file(files[0], mode='r')
        self.dim = np.array(h5.root.particles.time_0.read().tolist()).shape
        self.steps = len(np.array(h5.root.observation.read().tolist()))
        h5.close()


    def read(self, file):
        h5 = tables.open_file(file, mode='r')
        particles = np.array(getattr(h5.root.particles, 'time_' + str(self.steps-1)).read().tolist())
        h5.close()
        return particles


    def collect_mean_data(self):
      
        for i, file in enumerate(self.files):
            particles =  self.read(file)
            if i == 0:
                self.particles = particles
            else:
                self.particles += particles
     
        self.particles /= len(self.files)



    #"""
    def plot_line(self, folder, tag, obs_cov, dim, save=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)

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