# HALO

**Paper:** [Journal version](https://doi.org/10.1007/s10898-025-01555-9) | [arXiv preprint](https://arxiv.org/abs/2211.04129)

HALO (Hybrid Adaptive Lipschitzian Optimization) is a deterministic partition based global optimization algorithm for box constrained black box problems of the form

```math
\min_{x} f(x)
\quad \text{subject to} \quad x \in D,
```

where

```math
D = \left\{x \in \mathbb{R}^N : l \leq x \leq u \right\}.
```

The method combines partition based global search with adaptive estimates of the local Lipschitz constants to compute lower bounds and drive the algorithm towards global minimizers. HALO is coupled with local optimization routines from the `scipy.optimize` library to speed up convergence, and it provides a variable importance ranking to assist problem interpretation based on the approximate gradients collected during the search.

## Notation

The feasible set is denoted by $D \subset \mathbb{R}^N$, where $N$ is the dimension of the problem.

At iteration $k$, the domain is partitioned into subregions indexed by the set $I_k$. For each index $i_k \in I_k$, the corresponding partition is denoted by $D_{i_k}$.

The centroid of partition $D_{i_k}$ is denoted by $x_{i_k}$, while $l_{i_k}$ and $u_{i_k}$ denote its lower and upper bounds, respectively.

HALO builds an approximate gradient $\widetilde{\nabla} f(x_{i_k})$ around each centroid $x_{i_k}$ using only the points sampled so far by the algorithm. This quantity is used to estimate the local behavior of the objective function inside each partition.

## Adaptive estimate of the local Lipschitz constants

In Lipschitz optimization, lower bounds are used to guide the search toward regions that are more likely to contain a global minimizer. A single global estimate, however, may fail to capture the local behavior of the objective function inside a specific partition. HALO addresses this point by associating an adaptive local Lipschitz estimate with each partition.

The adaptive estimate used by HALO is

```math
\widetilde{L}_{i_k}
=
\alpha_{i_k}\widetilde{L}_k
+
(1-\alpha_{i_k})\left\|\widetilde{\nabla} f(x_{i_k})\right\|.
```

Here, $\alpha_{i_k}$ is a weight that determines how much the estimate relies on global information versus local information inside partition $D_{i_k}$. It is defined as

```math
\alpha_{i_k} = \frac{\left\|u_{i_k} - l_{i_k}\right\|}{\sqrt{N}},
\qquad
\alpha_{i_k} \in (0,1),
```

so that larger partitions receive a larger weight on the global term, while smaller partitions place more emphasis on the local term.

The quantity $\widetilde{L}_k$ is the global Lipschitz estimate at iteration $k$, defined as

```math
\widetilde{L}_k = \max_{i_k \in I_k} \left\|\widetilde{\nabla} f(x_{i_k})\right\|.
```

In other words, $\widetilde{L}_k$ is the largest approximate gradient norm observed over the current set of partitions.

This estimate is the core mechanism of HALO. When a partition is large, the local information around its centroid is less reliable, so the estimate places more emphasis on the global Lipschitz estimate at iteration $k$. When a partition becomes smaller, the approximation around its centroid becomes more informative, so the estimate places more emphasis on the norm of the approximate gradient evaluated at that centroid.
In this way, HALO balances global and local information in a self adaptive manner.

## Lower bounds and partition selection

Given $\widetilde{L}_{i_k}$, HALO computes a lower bound for each partition. These lower bounds are then used to decide which regions should be explored next.

The selection strategy is based on three simple ideas.

1. HALO selects the partition with the smallest lower bound, since this is the partition that appears most promising from a global optimization perspective.

2. HALO always keeps track of the partition containing the current best objective value. This ensures that the search does not lose focus on the best solution found so far.

3. HALO also considers the largest partitions and selects the most promising one among them according to the lower bound. This preserves exploration and supports the global search by preventing the algorithm from focusing too early only on very small regions.

This combination allows HALO to balance exploitation of promising regions with continued exploration of the domain.

## Adaptive gradient approximation

HALO does not assume that exact gradients are available.

Instead, it builds $\widetilde{\nabla} f(x_{i_k})$ from the points sampled during the partitioning process. These approximate gradients are used both to construct $\widetilde{L}_{i_k}$ and to extract information about the sensitivity of the objective function.

This means that the search itself produces the information needed for both optimization and interpretation.

## Hybrid local optimization

HALO can be coupled with local optimization routines from `scipy.optimize.minimize`. The currently supported methods are `L-BFGS-B`, `Nelder-Mead`, `TNC`, and `Powell`.

The local optimizer is used only as a refinement step. It does not replace the global partition based search.

A local search is triggered when a selected partition satisfies the selection conditions described above and when

```math
\frac{\left\|u_{i_k} - l_{i_k}\right\|}{2} \leq \beta.
```

The parameter $\beta$ only controls when local refinement is allowed to start. It is not part of the adaptive local Lipschitz estimate.

In HALO, local optimization is used as a refinement near promising centroids rather than as the main search mechanism. For this reason, the algorithm is robust with respect to $\beta$. The local search helps speed up convergence, while the global behavior of the method is still driven by the partitioning strategy and the lower bounds induced by $\widetilde{L}_{i_k}$.

## Variable importance and interpretability

At the end of the search, HALO can rank the variables according to the information collected in the matrix $G$, which stores the approximate gradients $\widetilde{\nabla} f(x_{i_k})$.

The variable importance is computed as

```math
\text{Variable Importance}
=
\frac{1}{|I_k|}
\sum_{i=1}^{|I_k|}
\widetilde{\nabla} f(x_{i_k}).
```

A final normalization step is then applied so that the variable importance vector sums to one.

This provides an interpretable summary of the directions in which the objective function has shown the highest sensitivity during the optimization process. As a result, HALO is not only a global optimization method, but also a tool that can help identify which variables matter most in the problem under study.

## Installation

Clone the repository:

```bash
git clone https://github.com/dannyzx/HALO.git
cd HALO
```

## Citation

If you use HALO in your work, please cite:

```bibtex
@article{d2026efficient,
  title={An efficient global optimization algorithm with adaptive estimates of the local Lipschitz constants},
  author={D’Agostino, Danny},
  journal={Journal of Global Optimization},
  pages={1--32},
  year={2026},
  publisher={Springer}
}
```
