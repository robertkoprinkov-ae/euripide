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

class BoundedGP():

    def __init__(self, N_hat, x_min, x_max, k, interpol_x=None, interpol_y=None, lowerbound=None, upperbound=None):
        self.hats = hats(N_hat, x_min, x_max)

        self.cov_prior = np.zeros((self.hats.n_splines, self.hats.n_splines))
        self.mean_prior = np.zeros(self.hats.n_splines)

        for i in range(self.hats.n_splines):
            for j in range(self.hats.n_splines):
                self.cov_prior[i, j] = k(self.hats.hat_x[i], self.hats.hat_x[j])

        self.interpol_x = interpol_x
        self.interpol_y = interpol_y

        self.mean_interpol = None
        self.cov_interpol = None

        if self.interpol_x is not None:
            A = self.hats.eval_hats(interpol_x)
            self.mean_interpol = (A @ self.cov_prior).T @ np.linalg.solve(A @ self.cov_prior @ A.T, interpol_y)
            print('Cond', np.linalg.cond(np.linalg.inv(A @ self.cov_prior @ A.T)))
            self.cov_interpol  = self.cov_prior - (A @ self.cov_prior).T @ np.linalg.inv(A @ self.cov_prior @ A.T) @ A @ self.cov_prior
            # why is the condition horrible?
            print(np.linalg.cond((A @ self.cov_prior).T @ np.linalg.inv(A @ self.cov_prior @ A.T) @ (A @ self.cov_prior)))
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if self.lowerbound is not None:
            A = self.hats.eval_hats(interpol_x)
            interpolation_constraint = scipy.optimize.LinearConstraint(A, interpol_y, interpol_y)
            # why cov_prior and not cov_interpol?
            # it doesn't matter much, just improves the rejection sampling %
            # once rejection sampling works see percentage with and without using cov_prior
            optim_result = scipy.optimize.minimize(lambda x: x.T @ self.cov_interpol @ x, np.zeros_like(self.hats.hat_x), 
                                                            method='COBYQA', bounds=((lowerbound, upperbound),), 
                                                            constraints=interpolation_constraint)
            print(optim_result)
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

        n_accept = 0

        for n_iter in range(10*n_samples):
            potential = np.random.multivariate_normal(self.mean_constrained, self.cov_interpol, n_samples)
            unif = np.random.uniform(size=n_samples)
            # doesn't work. Why would this work?
            # why are these samples still interpolated?
            accept = unif < np.exp(self.mean_constrained.T @ np.linalg.inv(self.cov_prior) @ self.mean_constrained - 
                                   potential @ np.linalg.inv(self.cov_prior) @ self.mean_constrained)
            n_newly_accepted = min(accept.sum(), n_samples-n_accept)
            samples[n_accept:n_accept+n_newly_accepted, :] = potential[accept, :][:n_newly_accepted]

            n_accept += n_newly_accepted

            print('Accepted: %d; %.2f percent' % (n_newly_accepted, n_newly_accepted/n_samples))
            if n_accept == n_samples:
                break
        return samples
