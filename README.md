# HALO
## Introduction
HALO is a derivative-free deterministic global optimization algorithm designed for solving for black-box problems.<br />
You can find the algorithm description inside the preprint at the following [link](https://arxiv.org/abs/2211.04129).<br />
HALO stands for Hybrid Adaptive Lipschitzian Optimization.
## Installation
Clone the repo:
   ```sh
   git clone https://github.com/dannyzx/HALO.git
   ```
## Example
   ```python
from halo import HALO
import math

bounds = [[-5., 10.], [0., 15.]] # bounds is a list or numpy array where in the
# first column there are the lower bounds
# and in the second column the upper bounds
# e.g. [[x_min., x_max], [y_min, y_max]]

def fun(x):  # Branin function
    return ((x[1] - (5.1 / (4 * math.pi ** 2)) * x[0] ** 2
             + 5 * x[0] / math.pi - 6) ** 2
            + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x[0]) + 10)

max_feval = 110  # maximum number of function evaluations
max_iter = 1000  # maximum number of iterations
beta = 1e-3  # beta controls the usage of the local optimizers during the optimization process
# With a lower value of beta HALO will use the local search more rarely and viceversa.
# The parameter beta must be less than or equal to 1e-2 or greater than equal to 1e-4.
local_optimizer = 'L-BFGS-B' # Choice of local optimizer from scipy python library.
# The following optimizers are available: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.
# For more infomation about the local optimizers please refer the scipy documentation.
verbose = 1  # this controls the verbosity level, fixed to 0 no output of the optimization progress 
# will be printed.

halo = HALO(fun, bounds, max_feval, max_iter, beta, local_optimizer, verbose)
results = halo.minimize()
# results is a dictionary where each key value pair contains a particular information 
# about the optimization process carried out by HALO. 
# For example, with the keys 'best_x', 'best_f' and 'best_feval' we can access 
# the best decision variable, its function value, and the number 
# of function evaluations when it has been obtained.
best_x, best_f, best_feval = results['best_x'],  results['best_f'],  results['best_feval']
   ```

## How to cite
If you use this algorithm for your research please cite the following work:

```tex
@article{D_Agostino_An_Efficient_Global_2022,
author = {D'Agostino, Danny},
doi = {https://doi.org/10.48550/arXiv.2211.04129},
journal = {ArXiv},
title = {{An Efficient Global Optimization Algorithm with Adaptive Estimates of the Local Lipschitz Constants}},
url = {https://arxiv.org/abs/2211.04129},
year = {2022}
}
```
