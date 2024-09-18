import src.BoundedGP as BoundedGP
import numpy as np
import src.vis as vis


interpol_x = np.array([0.1, 0.2, 0.3])
interpol_y = np.array([0.1, 0.2, 0.3])

lowerbound = 0.5
upperbound = 2.

def k(x, x_):
    sigma = 0.2
    l = 1.
    return np.power(sigma, 2.) * np.exp(-np.power(x-x_, 2.)/(2.*np.power(l, 2.)))

interpol_x, interpol_y = np.array([0.15, 0.55]), np.array([1.0, 0.667])

# this is going to be a problem, the knots. We will need an adaptive approach because we don't know a priori
# how the knots will be distributed. Maybe even something like multigrid.
boundedGP = BoundedGP.BoundedGP(10, 0., 1., k, interpol_x, interpol_y, lowerbound=lowerbound, upperbound=upperbound)

mean_knots, var_knots = boundedGP.statistics_prior()

#sample = boundedGP.sample_interpolation(sample_pts)

mean, var = boundedGP.statistics_interpolation()

print(boundedGP.sample_constrained(n_samples=5))

print(np.diag(boundedGP.cov_interpol), np.linalg.cond(boundedGP.cov_prior), np.linalg.cond(boundedGP.cov_interpol))

visualize = vis.Visualize()

visualize.plot_interpolated(boundedGP)
visualize.plot_constrained(boundedGP)
