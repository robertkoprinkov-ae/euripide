import matplotlib.pyplot as plt
import numpy as np

import src.BoundedGP as BoundedGP

splines = BoundedGP.monotonic_splines(4, 0., 1.)

fig, ax = plt.subplots(1, 1, figsize=(5, 2))

x = np.linspace(0., 1., 1000)

Y = splines.eval_splines(x)

for i in range(Y.shape[1]):
    ax.plot(x, Y[:, i], label=r'$\phi_%d$' % i)
ax.legend()
ax.grid()

plt.savefig('img/monotonic_splines.png', dpi=300)
