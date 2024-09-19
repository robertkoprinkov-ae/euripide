import src.BoundedGP as BoundedGP
import numpy as np
import src.vis as vis


lowerbound = -10.
upperbound = 20.

# Gaussian kernel
def k(x, x_):
    sigma = 25.
    theta = 0.2
    return np.power(sigma, 2.) * np.exp(-np.power(x-x_, 2.)/(2*np.power(theta, 2.)))

interpol_x, interpol_y = np.array([0.0, 0.1, 0.19, 0.41, 0.52, 0.9]), np.array([10.0, 8., -9., -6., 12., 17.])

# this is going to be a problem, the knots. We will need an adaptive approach because we don't know a priori
# how the knots will be distributed. Maybe even something like multigrid.
boundedGP = BoundedGP.BoundedGP(50, 0., 1., k, interpol_x, interpol_y, lowerbound=lowerbound, upperbound=upperbound)

mean_knots, var_knots = boundedGP.statistics_prior()

#sample = boundedGP.sample_interpolation(sample_pts)

mean, var = boundedGP.statistics_interpolation()

visualize = vis.Visualize()

visualize.plot_interpolated(boundedGP)
visualize.plot_constrained(boundedGP)
