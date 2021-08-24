# User Guide

## Setup
SparseGPs builds on top of [AbstractGPs.jl](https://juliagaussianprocesses.github.io/AbstractGPs.jl/dev/), so all of its features are reexported automatically by SparseGPs.
```julia
using SparseGPs, Random
rnd = MersenneTwister(1453)  # set a random seed
```

First, we construct a prior Gaussian process with a Matern-3/2 kernel and zero mean function, and sample some data. More exotic kernels can be constructed using [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/userguide/).
```julia
f = GP(Matern32Kernel())

x = rand(rng, 100)
fx = f(x, 0.1)  # Observe the GP with Gaussian observation noise (σ² = 0.1)
y = rand(rng, f(x))  # Sample from the GP prior at x
```

### The exact GP posterior
The exact posterior of `f` conditioned on `y` at inputs `x` is given by
```julia
exact_posterior = posterior(fx, y)
```

## Constructing a sparse approximation
To construct a sparse approximation to the exact posterior, we first need to select some inducing inputs. In this case, we simply pick a subset of the training data, but more sophisticated schemes for inducing point selection are provided in [InducingPoints.jl](https://juliagaussianprocesses.github.io/InducingPoints.jl/stable/).
```julia
M = 15  # The number of inducing points
z = x[1:M]
```
The inducing inputs `z` imply some latent function values `u = f(z)`, sometimes called pseudo-points. The stochastic variational Gaussian process (SVGP) approximation is defined by a variational distribution `q(u)` over the pseudo-points. In the case of GP regression, the optimal form for `q(u)` is a multivariate Gaussian, which is the only form of `q` currently supported by this package.
```julia
using Distributions, LinearAlgebra
q = MvNormal(zeros(length(z)), I)
```
Finally, we pass our `q` along with the inputs `f(z)` to obtain an approximate posterior GP:
```julia
fz = f(z, 1e-6)  # 'observe' the process at z with some jitter for numerical stability 
approx = SVGP(fz, q)  # Instantiate everything needed for the svgp approximation

svgp_posterior = posterior(approx)  # Create the approximate posterior
```

## The Evidence Lower Bound (ELBO)
The approximate posterior constructed above will be a very poor approximation, since `q` was simply chosen to have zero mean and covariance `I`. A measure of the quality of the approximation is given by the ELBO. Optimising this term with respect to the parameters of `q` and the inducing input locations `z` will improve the approximation.
```julia
elbo(SVGP(fz, q), fx, y)
```
A detailed example of how to carry out such optimisation is given in [1. Regression: Sparse Variational Gaussian Process for Stochastic Optimisation with Flux.jl](@ref). For an example of non-conjugate inference, see [2. Classification: Sparse Variational Approximation for Non-Conjugate Likelihoods with Optim's L-BFGS](@ref).
