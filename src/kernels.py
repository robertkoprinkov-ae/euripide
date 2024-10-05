import numpy as np

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


