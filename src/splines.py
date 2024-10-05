import numpy as np

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

