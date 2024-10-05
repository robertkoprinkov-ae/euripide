import unittest
from src.splines import hats as hats
from src.splines import monotonic_splines as monotonic_splines

from src.BoundedGP import BoundedGP
from src.kernels import gaussian_kernel

from src.vis import Visualize

import numpy as np

import matplotlib.pyplot as plt
import scienceplots

import functools

plt.style.use(['science', 'ieee'])

class TestInterpolationStats(unittest.TestCase):
    def setUp(self):
        self.vis = Visualize()
        
    def testCovarianceDimension(self):

        n_samples = [i for i in range(30)]
        
        dim_posterior = np.zeros(len(n_samples))
        for i, n in enumerate(n_samples):
            interpol_x = np.linspace(0., 1., n)
            interpol_y = np.zeros_like(interpol_x)
            bgp = BoundedGP(50, 0., 1., gaussian_kernel({'theta': 0.1, 'sigma': 1.}), interpol_x, interpol_y, nugget_prior=1e-3, nugget_interpol=None)

            _, cov_posterior = bgp.statistics_interpolation()
            
            eigvals = np.linalg.eigvals(cov_posterior)
            dim_posterior[i] = (eigvals > 1e-9).sum()
        
        fig, ax = plt.subplots(3, 1, figsize=(5, 6))
        ax[0].plot(n_samples, dim_posterior, label=r'Dim. of $\Sigma$')

        ax[0].set_xlabel('\# interpolation points')
        ax[0].set_ylabel(r'Dimensionality of posterior $\Sigma$')
        
        ax[0].set_title(r'Number of eigenvalues $\geq$ 1e-9')
        ax[0].grid()
        ax[0].legend()
        
        for i, n in enumerate([15, 29]):
            interpol_x = np.linspace(0., 1., n)
            interpol_y = np.zeros_like(interpol_x)
            bgp = BoundedGP(50, 0., 1., gaussian_kernel({'theta': 0.1, 'sigma': 1.}), interpol_x, interpol_y, nugget_prior=1e-3, nugget_interpol=None)

            _, cov_posterior = bgp.statistics_interpolation()
            
            eigvals = np.linalg.eigvals(cov_posterior)
            ax[i+1].hist(eigvals)
            ax[i+1].set_xlabel(r'Eigenvalue $\lambda_i$')
            ax[i+1].set_ylabel('Occurrence \#')

            ax[i+1].set_title('Eigenvalue distribution when interpolating on %d points' % n)

            ax[i+1].set_xlim(0.0, 0.005)
        plt.tight_layout()
        plt.savefig('img/test_distribution_posterior_cov_dimension.png', dpi=300)
