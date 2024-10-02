import numpy as np
import scipy
from qpsolvers import solve_qp

class spline():
    def eval_spline(self, spline_i, x_sample):
        pass

    """
        Evaluate each of the spline functions at the points in the array x, 
        i.e. returns Phi(x_eval)^T = [--phi(x_0)^T--, --phi(x_1)^T--, --phi(x_2)^T--, ..., --phi(x_{n_eval})^T--]

        eval_splines(interpol_x) calculates the matrix A from the paper
    
        @param np.array of dimension (n_eval_pts)
        @return np.array of dimension (n_eval_pts, self.n_splines)
        
    """    
    def eval_splines(self, x_eval):
        A = np.zeros((x_eval.shape[0], self.n_splines))
    
        for j in range(self.n_splines):
            A[:, j] = self.eval_spline(j, x_eval)
        return A
    
class hats(spline):

    def __init__(self, N_splines, x_min, x_max):
        self.splines_x = np.linspace(x_min, x_max, N_splines+1)
        self.dx_splines = self.splines_x[1]-self.splines_x[0]
        self.n_splines = N_splines+1
    """
        Evaluate the spline_ith hat function at the points x_sample, i.e. phi_i(X_sample) = [phi_i(x_0), phi_i(x_1), ...]
        
        @param np.array of dimension (n_eval_pts)
        @return np.array of dimension (n_eval_pts)
    """
    def eval_spline(self, spline_i, x_sample):
        x_spline_ = self.splines_x[spline_i]
        Y = np.where(x_sample < x_spline_ + 1e-9, np.maximum(0., 1.-(x_spline_-x_sample)/self.dx_splines), np.maximum(0., 1.-(x_sample-x_spline_)/self.dx_splines))
        return Y
    
class monotonic_splines(spline):

    def __init__(self, N_splines, x_min, x_max):
        self.splines_x = np.linspace(x_min, x_max, N_splines+1)
        self.dx_splines = self.splines_x[1]-self.splines_x[0]
        self.n_splines = N_splines+2
    
    """
        Evaluate the spline_ith spline function at the points x_sample
    """
    def eval_spline(self, spline_i, x_sample):
        if spline_i == 0:
            # constant function, but is this correct?
            return 1.
        x_spline_ = self.splines_x[spline_i-1]
        if spline_i == 1:
            # these expressions are the same as below, but the first half of the spline is missing,
            # so the integral of the second half is also less
            Y = np.zeros_like(x_sample)
            Y = np.where(np.logical_and(x_sample > x_spline_, x_sample < x_spline_ + self.dx_splines + 1e-9),
                         x_sample - np.power(x_sample, 2.)/(2*self.dx_splines) + (x_sample * x_spline_)/self.dx_splines + 0. - x_spline_ - np.power(x_spline_, 2.)/(2*self.dx_splines), Y)
            Y = np.where(x_sample > x_spline_ + self.dx_splines, self.dx_splines/2., Y)
        else:

            Y = np.zeros_like(x_sample)
            Y = np.where(np.logical_and(x_sample > x_spline_ - self.dx_splines - 1e-9, x_sample < x_spline_ + 1e-9), 
                         np.power(x_sample, 2.)/(2*self.dx_splines) + ((self.dx_splines - x_spline_)*x_sample)/self.dx_splines + np.power(x_spline_ - self.dx_splines, 2.)/(2*self.dx_splines), Y)
            Y = np.where(np.logical_and(x_sample > x_spline_, x_sample < x_spline_ + self.dx_splines + 1e-9),
                         x_sample - np.power(x_sample, 2.)/(2*self.dx_splines) + (x_sample * x_spline_)/self.dx_splines + self.dx_splines/2. - x_spline_ - np.power(x_spline_, 2.)/(2*self.dx_splines), Y)
            Y = np.where(x_sample > x_spline_ + self.dx_splines, self.dx_splines, Y)
        return Y

class kernel():

    """
        Initialize kernel with parameters

        @params params Dictionary. Key value pairs. 
    """
    def __init__(self, params):
        pass

    """
        Kernel function evaluated at points x and x_ K(x, x_) 
    """
    def k(self, x, x_):
        pass

    """
       Derivative w.r.t. first variable.
    """
    def k_der_x1(self, x, x_):
        pass

    """
        Derivative w.r.t. second variable. Assuming symmetric kernel, so equivalent to self.k_der_x1(x_, x) 
    """
    def k_der_x2(self, x, x_):
        return self.k_der_x1(x_, x)

    """
        Second derivative of kernel, w.r.t. first and second. Assuming symmetric kernel, so order of derivation does not matter. 
    """
    def k_der_x1x2(self, x, x_):
        pass
    
class gaussian_kernel(kernel):

    """
        Initialize kernel with parameters

        @params params Dictionary. Key value pairs. 
    """
    def __init__(self, params):
        self.theta = params['theta']
        self.sigma = params['sigma']

    """
        Kernel function evaluated at points x and x_ K(x, x_) 
    """
    def k(self, x, x_):
        return np.power(self.sigma, 2.) * np.exp(-np.power(x-x_, 2.)/(2*np.power(self.theta, 2.)))        

    """
       Derivative w.r.t. first variable.
    """
    def k_der_x1(self, x, x_):
        return self.k(x, x_) * -(x-x_)/np.power(self.theta, 2.)

    """
        Derivative w.r.t. second variable. Assuming symmetric kernel, so equivalent to self.k_der_x1(x_, x) 
    """
    def k_der_x2(self, x, x_):
        return self.k(x, x_) * (x-x_)/np.power(self.theta, 2.)
        # indeed equal to self.k_der_x1(x_, x)
    
    """
        Second derivative of kernel, w.r.t. first and second. Assuming symmetric kernel, so order of derivation does not matter. 
    """
    def k_der_x1x2(self, x, x_):
        return self.k(x, x_) * (1/np.power(self.theta, 2.) - np.power(x-x_, 2.)/np.power(self.theta, 4.))

"""
    Abstract class for constrained Gaussian Processes. Monotonic and Bounded GPs inherit from this class.
    
"""
class constrainedGP():
    """
        Calculate distribution of RV $\\xi$ given the interpolation points, with the prior belonging to this GP.    
    """
    def calc_interpolation(self, interpol_x, interpol_y, nugget=5e-3):
        A = self.splines.eval_splines(interpol_x)
        
        mean_interpol = (A @ self.cov_prior).T @ np.linalg.solve(A @ self.cov_prior @ A.T, interpol_y)
        cov_interpol  = self.cov_prior - (A @ self.cov_prior).T @ np.linalg.inv(A @ self.cov_prior @ A.T) @ A @ self.cov_prior

        print('[PosDef] Post-interpolation matrix:', np.all(np.linalg.eigvals(cov_interpol) > 0))
        print('[PosDef] Prior covariance matrix:', np.all(np.linalg.eigvals(self.cov_prior) > 0))

        if nugget is not None:
            cov_noise = cov_interpol + 5e-3 * np.eye(self.splines.n_splines)# np.random.uniform(size=self.cov_interpol.shape)

            print('[Nugget] added to post-interpolation covariance matrix. Condition number went from %d to %d.' % (np.linalg.cond(cov_interpol), np.linalg.cond(cov_noise)))
            print('[PosDef] Post-interpolation covariance matrix with nugget:', np.all(np.linalg.eigvals(cov_noise) > 0))

            # adding noise does decrease it
            cov_interpol = cov_noise

        return mean_interpol, cov_interpol

    """
        Get mean and covariance of used RV, or, if points are supplied, of the RVs Y^{N}(x_0, x_1, ..., x_n)
    """
    def statistics_prior(self, x=None):
        if x is None: # return mean and cov. matrix of RV used in finite dimensional representation of GP
            return self.mean_prior, self.cov_prior
        else:
            phi_T = self.spline.eval_splines(x) # I believe this expression is correct, but it should be tested, e.g. for hat function splines
            return phi_T @ self.mean_prior, phi_T @ self.cov_prior @ phi_T.T

    """
        Sample from prior distribution
        
    """
    def sample_prior(self, x_sample=None, n_samples=1):
        if x_sample is None:
            x_sample = self.splines.splines_x
        RV = np.random.multivariate_normal(self.mean, self.cov, n_samples).T # dim: self.splines.n_splines x n_samples
        phi_T = self.splines.eval_splines(x_sample)

        return phi_T @ RV

    """
        Mean and variance of the distribution of RV used in representation, obtained by conditioning the prior distribution on the interpolation points.
        This distribution is normal.

        Alternatively, if x is supplied, the mean and covariance of the RV given by Y^{N}(x_0), Y^{N}(x_1), ...Y^N(x_{n_x})|Y(x_I) = Y_I is returned.
        
    """
    def statistics_interpolation(self, x=None):
        if x is None:
            return self.mean_interpol, self.cov_interpol
        else:
            phi_T = self.splines.eval_splines(x)
            return phi_T @ self.mean_interpol, phi_T @ self.cov_interpol @ phi_T.T

    """
        Sample from distribution obtained by conditioning the prior distribution on the interpolation points. Note that this
        is not the constrained distribution, and that therefore the samples are not guaranteed to satisfy the desired constraints.
        
    """
    def sample_interpolation(self, x_sample=None, n_samples=1):
        if x_sample is None:
            x_sample = self.splines.splines_x
        assert self.mean_interpol is not None and self.cov_interpol is not None
        RV = np.random.multivariate_normal(self.mean_interpol, self.cov_interpol, n_samples) # dim: self.splines.n_splines x n_samples
        phi_T = self.splines.eval_splines(x_sample)
        return RV @ phi_T.T

    def statistics_constrained(self, x=None):
        # hard. only sending mean
        if x is None: # sending mean of RV used in representation of GP
            return self.mean_interpol, None
        else:
            phi_T = self.splines.eval_splines(x)
            return phi_T @ self.mean_constrained, None

    """
        x_sample: points x at which to return the RV Y^N(x). If None the RV corresponding to the underlying finite-dimensional
                  representation are returned
    """
    def sample_constrained(self, x_sample=None, n_samples=1):
        pass

    """
        Evaluate a given realization of the RV xi at new points x_eval
        
    """
    def eval_sample(self, x_eval, RV):
        phi_T = self.splines.eval_splines(x_eval)
        return RV @ phi_T.T
        
class BoundedGP(constrainedGP):

    def bounded_GP_prior(self, nugget=1e-3):
        cov_prior = np.zeros((self.splines.n_splines, self.splines.n_splines))
        mean_prior = np.zeros(self.splines.n_splines)        
        for i in range(self.splines.n_splines):
            for j in range(self.splines.n_splines):
                cov_prior[i, j] = self.kernel.k(self.splines.splines_x[i], self.splines.splines_x[j])

        if nugget is not None:
            old_cond = np.linalg.cond(cov_prior)
            cov_prior = cov_prior + nugget*np.eye(self.splines.n_splines)
            print('[Nugget] added to prior covariance matrix. Condition number went from %d to %d.' % (old_cond, np.linalg.cond(cov_prior)))
        return mean_prior, cov_prior
    
    def __init__(self, N_splines, x_min, x_max, kernel, interpol_x=None, interpol_y=None, lowerbound=None, upperbound=None):
        self.splines = hats(N_splines, x_min, x_max)
        self.kernel = kernel
        
        self.mean_prior, self.cov_prior = self.bounded_GP_prior(nugget=1e-3)
        
        self.interpol_x = interpol_x
        self.interpol_y = interpol_y

        self.mean_interpol = None
        self.cov_interpol = None

        if self.interpol_x is not None:
            self.mean_interpol, self.cov_interpol = self.calc_interpolation(interpol_x, interpol_y, nugget=1e-3)
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if self.lowerbound is not None:
            A = self.splines.eval_splines(interpol_x)
            interpolation_constraint = scipy.optimize.LinearConstraint(A, interpol_y, interpol_y)
            
            constraint_matrix = np.zeros((2*(self.splines.n_splines), self.splines.n_splines))
            constraint_matrix[:self.splines.n_splines, :] = np.eye(self.splines.n_splines)
            constraint_matrix[self.splines.n_splines:, :] = -np.eye(self.splines.n_splines)

            constraints_max_value = np.zeros(2*(self.splines.n_splines))
            constraints_max_value[:self.splines.n_splines] = upperbound
            constraints_max_value[self.splines.n_splines:] = -lowerbound

            # whether we optimize w.r.t. \Gamma^{N} or \Sigma doesn't matter. The minimum is equal in our case (I believe),
            # but definitely not in the general case. The prior matrix is better conditioned, so we will use that one here.
            #optim_result_no_bounds = solve_qp(np.linalg.inv(self.cov_interpol), -self.mean_interpol.T @ np.linalg.inv(self.cov_interpol), constraint_matrix, constraints_max_value, None, None, lb=constraints[:, 0], ub=constraints[:, 1], solver='quadprog', verbose=True)
            optim_result = solve_qp(np.linalg.inv(self.cov_interpol), -self.mean_interpol.T @ np.linalg.inv(self.cov_interpol), constraint_matrix, constraints_max_value, None, None, solver='quadprog', verbose=True)

            # in extreme cases, the prior solver works better
            # for some reason, the posterior one doesn't work in the monotonic case where we have an almost constant stretch
            # the constraint ends up being violated
            optim_prior = solve_qp(np.linalg.inv(self.cov_prior), np.zeros(self.splines.n_splines), constraint_matrix, constraints_max_value, A, interpol_y, solver='quadprog', verbose=True)
            #optim_result = scipy.optimize.minimize(lambda x: (x-self.mean_interpol).T @ np.linalg.solve(self.cov_interpol, (x-self.mean_interpol)), np.zeros_like(self.mean_prior), 
            #                                                method='COBYQA', bounds=((lowerbound, upperbound),), 
            #                                                constraints=interpolation_constraint)
            #optim_prior = scipy.optimize.minimize(lambda x: x.T @ np.linalg.solve(self.cov_prior, x), np.zeros_like(self.mean_interpol), 
            #                                                method='COBYQA', bounds=((lowerbound, upperbound),), 
            #                                                constraints=interpolation_constraint)
            #assert optim_result.success, optim_result
            self.mean_constrained = optim_result
            self.mean_constrained_prior = optim_prior

    # verify that the below 4 functions give the same output now and before
    #def statistics_prior(self):
    #    return self.mean_prior, np.diag(self.cov_prior)
        
    #def sample_prior(self, n_samples=1):
    #    return np.random.multivariate_normal(self.mean, self.cov, n_samples).T

    #def statistics_interpolation(self):
    #    return self.mean_interpol, np.diag(self.cov_interpol)
    
    #def sample_interpolation(self, n_samples=1):
    #    assert self.mean_interpol is not None and self.cov_interpol is not None
    #    return np.random.multivariate_normal(self.mean_interpol, self.cov_interpol, n_samples).T

    def sample_constrained(self, x_sample=None, n_samples = 1, return_stats=False, throw_exception_if_failed=True):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        assert self.mean_constrained is not None
        phi_T = None
        if x_sample is not None:
            phi_T = self.splines.eval_splines(x_sample)
        samples = np.zeros((n_samples, self.splines.n_splines))

        n_accepted = 0
        n_rejected = 0
        n_rejected_constraint = 0
        for n_iter in range(1000*n_samples):
            potential = np.random.multivariate_normal(self.mean_constrained, self.cov_interpol, n_samples)

            accepted_convex = np.logical_and(np.all(potential > self.lowerbound-1e-9, axis=1), np.all(potential < self.upperbound + 1e-9, axis=1))
            
            n_rejected_constraint += n_samples - accepted_convex.sum()
            
            unif = np.random.uniform(size=n_samples)
            # Have to use cov_interpol, otherwise Neumann theorem doesn't apply and we can't sample from the original distribution in this way.
            accepted_neumann_sampling = unif < np.exp((potential-self.mean_constrained.T) @ np.linalg.solve(self.cov_interpol, self.mean_interpol - self.mean_constrained))
            #accepted_neumann_sampling = unif < np.exp(self.mean_constrained.T @ np.linalg.solve(self.cov_interpol, self.mean_constrained) - 
            #                       potential @ np.linalg.solve(self.cov_interpol, self.mean_constrained))
            accepted = np.logical_and(accepted_convex, accepted_neumann_sampling)
            n_newly_accepted = min(accepted.sum(), n_samples-n_accepted)

            if x_sample is None:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted]
            else:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted] @ phi_T.T                
            n_accepted += accepted.sum()
            n_rejected += n_samples - accepted.sum()
            if n_accepted >= n_samples:
                break
        print('Sampling %.2f%% (accepted); %.2f%% (Rejected, violated constraints); %.2f%% (Rejected, Neumann sampling)' % (100.*n_accepted/(n_accepted+n_rejected),\
              100*n_rejected_constraint/(n_accepted+n_rejected), 100*(n_rejected-n_rejected_constraint)/(n_accepted+n_rejected)))
        if throw_exception_if_failed:
            assert n_accepted >= n_samples

        if return_stats:
            return samples[:n_accepted, :], {'n_rejected_constraint': n_rejected_constraint, 'n_rejected_neumann': n_rejected-n_rejected_constraint, 'n_accepted': n_accepted}
        else:
            return samples[:n_accepted, :]

class MonotonicGP(constrainedGP):

    def monotonic_prior(self, nugget=1e-3):
        mean_prior = np.zeros(self.splines.n_splines)
        cov_prior = np.zeros((self.splines.n_splines, self.splines.n_splines))

        cov_prior[0, 0] = self.kernel.k(0., 0.)

        for i in range(1, self.splines.n_splines):
            cov_prior[0, i] = self.kernel.k_der_x1(0., self.splines.splines_x[i-1])
            # symmetric kernel
            cov_prior[i, 0] = cov_prior[0, i]

        for i in range(1, self.splines.n_splines):
            for j in range(1, self.splines.n_splines):
                cov_prior[i, j] = self.kernel.k_der_x1x2(self.splines.splines_x[i-1], self.splines.splines_x[j-1])

        if nugget is not None:
            old_cond = np.linalg.cond(cov_prior)
            cov_prior = cov_prior + 1e-3*np.eye(self.splines.n_splines)
            print('[Nugget] added to prior covariance matrix. Condition number went from %d to %d.' % (old_cond, np.linalg.cond(cov_prior)))
            
        
        return mean_prior, cov_prior
    
    def __init__(self, N_splines, x_min, x_max, kernel, interpol_x=None, interpol_y=None, lowerbound=None, upperbound=None):
        self.splines = monotonic_splines(N_splines, x_min, x_max)

        self.kernel = kernel

        self.mean_prior, self.cov_prior = self.monotonic_prior(nugget=1e-6)
        
        self.interpol_x = interpol_x
        self.interpol_y = interpol_y

        self.mean_interpol = None
        self.cov_interpol = None
        
        if self.interpol_x is not None:
            self.mean_interpol, self.cov_interpol = self.calc_interpolation(interpol_x, interpol_y, nugget=1e-6)

        A = self.splines.eval_splines(interpol_x)
        interpolation_constraint = scipy.optimize.LinearConstraint(A, interpol_y, interpol_y)
        
        self.lowerbound, self.upperbound = lowerbound, upperbound

        constraints = [[-1e-5, 1e5]] + [[lowerbound, upperbound] for x in range(self.splines.n_splines-1)]
        constraints = np.array(constraints)

        constraint_matrix = np.zeros((2*(self.splines.n_splines-1), self.splines.n_splines))
        constraint_matrix[:self.splines.n_splines-1, 1:] = np.eye(self.splines.n_splines-1)
        constraint_matrix[self.splines.n_splines-1:, 1:] = -np.eye(self.splines.n_splines-1)

        constraints_max_value = np.zeros(2*(self.splines.n_splines-1))
        constraints_max_value[:self.splines.n_splines-1] = upperbound
        constraints_max_value[self.splines.n_splines-1:] = -lowerbound
        
        optim_result_no_bounds = solve_qp(np.linalg.inv(self.cov_interpol), -self.mean_interpol.T @ np.linalg.inv(self.cov_interpol), constraint_matrix, constraints_max_value, None, None, lb=constraints[:, 0], ub=constraints[:, 1], solver='quadprog', verbose=True)
        optim_result = solve_qp(np.linalg.inv(self.cov_interpol), -self.mean_interpol.T @ np.linalg.inv(self.cov_interpol), constraint_matrix, constraints_max_value, None, None, solver='quadprog', verbose=True)

        # in extreme cases, the prior solver works better
        # for some reason, the posterior one doesn't work in the monotonic case where we have an almost constant stretch
        # the constraint ends up being violated
        optim_prior = solve_qp(np.linalg.inv(self.cov_prior), np.zeros(self.splines.n_splines), constraint_matrix, constraints_max_value, A, interpol_y, solver='quadprog', verbose=True)

        print(optim_result.max(), optim_result.min())
        if optim_prior is None:
            print('Failed optim prior')
        #assert optim_result.success, optim_result
        self.mean_constrained = optim_result
        self.mean_constrained_prior = optim_prior

        # remove this
        #self.mean_constrained = optim_prior
        print('Optim posterior vs prior difference', np.abs(optim_result - optim_prior).max())

    """
        @param x_sample: points at which to return the sampled function. If x_sample is None, the original RV used to
                         represent this finite dimensional approximation to the GP are returned
    """
    def sample_constrained(self, x_sample=None, n_samples = 1, return_stats=False, throw_exception_if_failed=True):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        assert self.mean_constrained is not None

        phi_T = None
        if x_sample is not None:
            print(x_sample)
            phi_T = self.splines.eval_splines(x_sample)
        samples = np.zeros((n_samples, x_sample.shape[0]))
        n_accepted = 0
        n_rejected = 0
        n_rejected_constraint = 0
        for n_iter in range(100*n_samples):
            potential = np.random.multivariate_normal(self.mean_constrained, self.cov_interpol, n_samples)

            accepted_convex = np.all(np.logical_and(potential[:, 1:] > self.lowerbound, potential[:, 1:] < self.upperbound), axis=1)

            n_rejected_constraint += n_samples - accepted_convex.sum()
            unif = np.random.uniform(size=n_samples)
            # Have to use cov_interpol, otherwise Neumann theorem doesn't apply and we can't sample from the original distribution in this way.
            accepted_neumann_sampling = unif < np.exp((potential-self.mean_constrained.T) @ np.linalg.solve(self.cov_interpol, self.mean_interpol - self.mean_constrained))
            #accepted_neumann_sampling = unif < np.exp(self.mean_constrained.T @ np.linalg.solve(self.cov_interpol, self.mean_constrained) - 
            #                       potential @ np.linalg.solve(self.cov_interpol, self.mean_constrained))
            accepted = np.logical_and(accepted_convex, accepted_neumann_sampling)
            n_newly_accepted = min(accepted.sum(), n_samples-n_accepted)
            if x_sample is None:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted]
            else:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted] @ phi_T.T

            n_accepted += accepted.sum()
            n_rejected += n_samples - accepted.sum()
            if n_accepted >= n_samples:
                break
        print('Sampling %.2f%% (accepted); %.2f%% (Rejected, violated constraints); %.2f%% (Rejected, Neumann sampling)' % (100.*n_accepted/(n_accepted+n_rejected),\
              100*n_rejected_constraint/(n_accepted+n_rejected), 100*(n_rejected-n_rejected_constraint)/(n_accepted+n_rejected)))
        if throw_exception_if_failed:
            assert n_accepted >= n_samples

        if return_stats:
            return samples[:n_accepted, :], {'n_rejected_constraint': n_rejected_constraint, 'n_rejected_neumann': n_rejected-n_rejected_constraint, 'n_accepted': n_accepted}
        else:
            return samples[:n_accepted, :]
