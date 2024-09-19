import numpy as np
import scipy


class hats():
    def __init__(self, N_hat, x_min, x_max):
        self.hat_x = np.linspace(x_min, x_max, N_hat+1)
        self.dx_hat = self.hat_x[1]-self.hat_x[0]
        self.n_splines = N_hat+1

    """
        Evaluate the ith hat function at the points x_sample
        
        @param np.array of dimension (n_eval_pts)
        @return np.array of dimension (n_eval_pts)
    """
    def eval_hat(self, x_hat_, x_sample):
        Y = np.where(x_sample < x_hat_ + 1e-9, np.maximum(0., 1.-(x_hat_-x_sample)/self.dx_hat), np.maximum(0., 1.-(x_sample-x_hat_)/self.dx_hat))
        return Y
    
    """
        Evaluate each of the hat functions at the points in the array x

        eval_hats(interpol_x) calculates the matrix A from the paper
    
        @param np.array of dimension (n_eval_pts)
        @return np.array of dimension (N_hat+1, n_eval_pts)
        
    """    
    def eval_hats(self, x_eval):
        A = np.zeros((x_eval.shape[0], self.hat_x.shape[0]))
    
        for j, x_j in enumerate(self.hat_x):
            A[:, j] = self.eval_hat(x_j, x_eval)
        return A

def monotonic():
    def __init__(self, N_splines, x_min, x_max):
        self.spline_x = np.linspoace(x_min, x_max, N_splines+1)
        self.dx_spline = self.spline_x[1]-self.spline_x[0]
        self.n_splines = N_splines+1

    def eval_spline(self, x_spline_, x_sample):
        return Y
    

class BoundedGP():

    def __init__(self, N_hat, x_min, x_max, k, interpol_x=None, interpol_y=None, lowerbound=None, upperbound=None):
        self.hats = hats(N_hat, x_min, x_max)

        self.cov_prior = np.zeros((self.hats.n_splines, self.hats.n_splines))
        self.mean_prior = np.zeros(self.hats.n_splines)

        for i in range(self.hats.n_splines):
            for j in range(self.hats.n_splines):
                self.cov_prior[i, j] = k(self.hats.hat_x[i], self.hats.hat_x[j])

        if np.linalg.cond(self.cov_prior) > 1e6:
            old_cond = np.linalg.cond(self.cov_prior)
            self.cov_prior = self.cov_prior + 1e-6*np.eye(self.hats.n_splines)
            print('[Nugget] added to prior covariance matrix. Condition number went from %d to %d.' % (old_cond, np.linalg.cond(self.cov_prior)))
        self.interpol_x = interpol_x
        self.interpol_y = interpol_y

        self.mean_interpol = None
        self.cov_interpol = None

        if self.interpol_x is not None:
            A = self.hats.eval_hats(interpol_x)
            self.mean_interpol = (A @ self.cov_prior).T @ np.linalg.solve(A @ self.cov_prior @ A.T, interpol_y)
            self.cov_interpol  = self.cov_prior - (A @ self.cov_prior).T @ np.linalg.inv(A @ self.cov_prior @ A.T) @ A @ self.cov_prior

            cov_noise = self.cov_interpol + 5e-3 * np.eye(self.hats.n_splines)# np.random.uniform(size=self.cov_interpol.shape)

            print('[Nugget] added to post-interpolation covariance matrix. Condition number went from %d to %d.' % (np.linalg.cond(self.cov_interpol), np.linalg.cond(cov_noise)))
            print('[PosDef] Post-interpolation matrix:', np.all(np.linalg.eigvals(self.cov_interpol) > 0))
            print('[PosDef] Post-interpolation covariance matrix with nugget:', np.all(np.linalg.eigvals(cov_noise) > 0))
            print('[PosDef] Prior covariance matrix:', np.all(np.linalg.eigvals(self.cov_prior) > 0))

            # adding noise does decrease it
            self.cov_interpol = cov_noise
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if self.lowerbound is not None:
            A = self.hats.eval_hats(interpol_x)
            interpolation_constraint = scipy.optimize.LinearConstraint(A, interpol_y, interpol_y)

            # whether we optimize w.r.t. \Gamma^{N} or \Sigma doesn't matter. The minimum is equal in our case (I believe),
            # but definitely not in the general case. The prior matrix is better conditioned, so we will use that one here.
            optim_result = scipy.optimize.minimize(lambda x: (x-self.mean_interpol).T @ np.linalg.solve(self.cov_interpol, (x-self.mean_interpol)), np.zeros_like(self.mean_prior), 
                                                            method='COBYQA', bounds=((lowerbound, upperbound),), 
                                                            constraints=interpolation_constraint)
            assert optim_result.success, optim_result
            self.mean_constrained = optim_result.x

    def statistics_prior(self):
        return self.mean_prior, np.diag(self.cov_prior)
        
    def sample_prior(self, n_samples=1):
        return np.random.multivariate_normal(self.mean, self.cov, n_samples).T

    def statistics_interpolation(self):
        return self.mean_interpol, np.diag(self.cov_interpol)
    
    def sample_interpolation(self, n_samples=1):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        return np.random.multivariate_normal(self.mean_interpol, self.cov_interpol, n_samples).T


    def statistics_constrained(self):
        # eh, hard
        pass
    
    def sample_constrained(self, n_samples = 1):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        assert self.mean_constrained is not None

        samples = np.zeros((n_samples, self.hats.n_splines))

        n_accepted = 0
        n_rejected = 0
        for n_iter in range(1000*n_samples):
            potential = np.random.multivariate_normal(self.mean_constrained, self.cov_interpol, n_samples)

            accepted_convex = np.logical_and(np.all(potential > self.lowerbound-1e-9, axis=1), np.all(potential < self.upperbound + 1e-9, axis=1))
            
            
            unif = np.random.uniform(size=n_samples)
            # Have to use cov_interpol, otherwise Neumann theorem doesn't apply and we can't sample from the original distribution in this way.
            accepted_neumann_sampling = unif < np.exp(self.mean_constrained.T @ np.linalg.solve(self.cov_interpol, self.mean_constrained) - 
                                   potential @ np.linalg.solve(self.cov_interpol, self.mean_constrained))
            accepted = np.logical_and(accepted_convex, accepted_neumann_sampling)
            n_newly_accepted = min(accepted.sum(), n_samples-n_accepted)
            samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted]

            n_accepted += n_newly_accepted
            n_rejected += n_samples - n_newly_accepted
            if n_accepted == n_samples:
                break
        print('Accepted: %d; %.2f percent' % (n_accepted, 100.*n_accepted/(n_accepted+n_rejected)))

        assert n_accepted == n_samples
        return samples
