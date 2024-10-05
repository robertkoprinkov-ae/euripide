import unittest
from src.BoundedGP import hats as hats
from src.splines import monotonic_splines as monotonic_splines

import numpy as np

import matplotlib.pyplot as plt
import scienceplots

import functools

plt.style.use(['science', 'ieee'])

class TestSplines(unittest.TestCase):
    def setUp(self):
        self.hats = hats(5, 0., 1.)
        self.mono_splines = monotonic_splines(20, 0., 3.)

    def testHatsLine(self):
        pass
    
    def integrate_fnc_splines(self, f):
        x_plot = np.linspace(0., 3., self.mono_splines.n_splines-1)
        phi_T = self.mono_splines.eval_splines(x_plot)[:, 1:]
        
        deriv = f(x_plot)
        return x_plot, phi_T @ deriv
        
    def testMonotonicSplinesLine(self):
        fig, ax = plt.subplots(3, 1, figsize=(5, 15))
        
        # we should be able to use the monotonic splines for integration
        # assuming in each of the examples below that the integral will be 0 at the point 0

        x, y = self.integrate_fnc_splines(lambda x: np.ones_like(x)) 
        ax[0].plot(x, x, label='Exact')
        ax[0].plot(x, y, label='Predicted by splines', linestyle='--')

        x, y = self.integrate_fnc_splines(lambda x: x)
        ax[1].plot(x, .5 * np.power(x, 2.), label='Exact')
        ax[1].plot(x, y, label='Predicted by splines', linestyle='--')

        x, y = self.integrate_fnc_splines(lambda x: np.exp(x))
        ax[2].plot(x, np.exp(x)-1., label='Exact')
        ax[2].plot(x, y, label='Predicted by splines', linestyle='--')
        
        titles = ['Linear function', 'Quadratic function', 'Exponential function']
        for i in range(3):
            ax[i].legend()
            ax[i].grid()

            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_title(titles[i])

        plt.savefig('img/test_monotonic_splines_integration.png')

