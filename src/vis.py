import matplotlib.pyplot as plt
import numpy as np

class Visualize():

    def __init__(self, directory='./img/'):
        self.directory = directory

    def plot_interpolated(self, boundedGP, n_samples=3, filename='interpolated.png'):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])

        y1, = ax[0].plot(boundedGP.hats.hat_x, boundedGP.mean_interpol)
        y2, = ax[0].plot(boundedGP.hats.hat_x, boundedGP.mean_interpol - 2*np.diag(boundedGP.cov_interpol), color='gray', linestyle='--')
        y3, = ax[0].plot(boundedGP.hats.hat_x, boundedGP.mean_interpol + 2*np.diag(boundedGP.cov_interpol), color='gray', linestyle='--')
        
        y_sample = []
        for i in range(n_samples):
            y_sample.append(ax[0].plot(boundedGP.hats.hat_x, boundedGP.sample_interpolation(), color='red', linestyle='-', linewidth=0.5, alpha=0.3)[0])
        
        ax[0].scatter(boundedGP.interpol_x, boundedGP.interpol_y, marker='x', color='red')
        
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].grid()

        ax[1].legend([y1, y2, y3] + y_sample, [r'$\mu_I$', r'$\mu_I - 2\sigma_I$', r'$\mu_I + 2\sigma_I$', 'samples'])
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.directory + filename, dpi=300)
    
    def plot_constrained(self, boundedGP, n_samples=100, filename='constrained.png'):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])

        y1, = ax[0].plot(boundedGP.hats.hat_x, boundedGP.mean_constrained)
        y2, = ax[0].plot(boundedGP.hats.hat_x, boundedGP.mean_interpol, linestyle='--')
        
        y_sample = []
        samples = boundedGP.sample_constrained(n_samples=n_samples)
        for i in range(n_samples):
            y_sample.append(ax[0].plot(boundedGP.hats.hat_x, samples[i], color='red', linestyle='-', linewidth=0.2, alpha=0.1)[0])
        
        ax[0].scatter(boundedGP.interpol_x, boundedGP.interpol_y, marker='x', color='red')
        
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].grid()

        ax[1].legend([y1, y2] + y_sample, [r'$\mu_{C+I}$', r'$\mu_{I}$', 'samples'])
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.directory + filename, dpi=300)
