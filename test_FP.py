import matplotlib.pyplot as plt
import numpy as np

import scienceplots

from src.BoundedGP import MonotonicGP as MonotonicGP
from src.kernels import gaussian_kernel

from src.vis import Visualize

from tqdm import tqdm

plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots(1, 1, figsize=(5, 2))

def f(x):
    # derivative: 1-2(x-1). = 1 - (2x - 2) = 3-2x, between 3 and -1 on [0, 2]
    #return x + np.power(x-1, 2.)
    # derivative: 2 - 2x => -1 at x=0.5, 1 at x=1.
    return 2*x - np.power(x, 2.)

def FP(x_init, f, n_iter=1000):
    x_hist = np.zeros(n_iter)
    x_hist[0] = x_init
    
    for n in range(n_iter-1):
        x_hist[n+1] = f(x_hist[n]).item()
    last_samples = x_hist[-int(n_iter//10):]
    #if last_samples.max() - last_samples.min() > 1e-1:
    print(last_samples.max())
    return (last_samples.max() - last_samples.min()) < 1e-1

def success(gp, xi, n_FP_tried=5):
    x_init = 0.5+np.random.rand()
    
    success = False
    for i in range(n_FP_tried):
        success = success or FP(x_init, lambda x: gp.eval_sample(np.array([x]), xi))

        if success:
            break
    return success

def calc_success_rate(gp):
    samples = gp.sample_interpolation(n_samples=20)
    
    success_arr = np.zeros(samples.shape[0])
    for i in tqdm(range(samples.shape[0])):
        success_arr[i] = success(gp, samples[i, :])
    return success_arr.sum()/success_arr.shape[0]

x = np.linspace(0., 2., 1000)
plt.plot(x, f(x), label=r'$y=x-x^2$')
plt.plot(x, x, linestyle='--', label=r'$y=x$')

plt.scatter(np.array([1.]), np.array([1.]), marker='x', color='red', label='FP')

plt.grid()

sample_pts = np.linspace(0., 2., 4+2)[1:-1]

plt.scatter(sample_pts, f(sample_pts), marker='+', color='blue', label='GP samples')

plt.savefig('img_FP/plot.png', dpi=300)


gp = MonotonicGP(50, 0., 2., gaussian_kernel({'theta': 0.1, 'sigma': 1.}), interpol_x=sample_pts, interpol_y=f(sample_pts), lowerbound=-1., upperbound=1., nugget_prior=1e-2, nugget_interpol=5e-2, verbose=True)

#print('Success rate', calc_success_rate(gp))
print('Exact', [FP(0.5+np.random.rand(), f) for i in range(100)])
vis = Visualize(directory='./img_FP/')

fig, ax = vis.plot_interpolated(gp, filename=None, return_ax=True)

ax[0].plot(x, f(x), label=r'y=x-x^2 (exact)', color='blue', linewidth=2.)

plt.savefig('img_FP/interpolated.png', dpi=300)

fig, ax = vis.plot_constrained(gp, filename=None, return_ax=True, n_samples=5)

ax[0].plot(x, f(x), label=r'y=x-x^2 (exact)', color='blue', linewidth=2.)
plt.savefig('img_FP/constrained.png', dpi=300)
