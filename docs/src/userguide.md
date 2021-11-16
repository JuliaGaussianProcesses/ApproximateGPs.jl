# User Guide

## Setup
ApproximateGPs builds on top of [AbstractGPs.jl](https://juliagaussianprocesses.github.io/AbstractGPs.jl/dev/), so all of its features are reexported automatically by ApproximateGPs.
```julia
using ApproximateGPs, Random
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
The inducing inputs `z` imply some latent function values `u = f(z)`, sometimes called pseudo-points. The [`SparseVariationalApproximation`](@ref) specifies a distribution `q(u)` over the pseudo-points. In the case of GP regression, the optimal form for `q(u)` is a multivariate Gaussian, which is the only form of `q` currently supported by this package.
```julia
using Distributions, LinearAlgebra
q = MvNormal(zeros(length(z)), I)
```
Finally, we pass our `q` along with the inputs `f(z)` to obtain an approximate posterior GP:
```julia
fz = f(z, 1e-6)  # 'observe' the process at z with some jitter for numerical stability 
approx = SparseVariationalApproximation(fz, q)  # Instantiate everything needed for the approximation

sva_posterior = posterior(approx)  # Create the approximate posterior
```

## The Evidence Lower Bound (ELBO)
The approximate posterior constructed above will be a very poor approximation, since `q` was simply chosen to have zero mean and covariance `I`. A measure of the quality of the approximation is given by the ELBO. Optimising this term with respect to the parameters of `q` and the inducing input locations `z` will improve the approximation.
```julia
elbo(SparseVariationalApproximation(fz, q), fx, y)
```
A detailed example of how to carry out such optimisation is given in [Regression: Sparse Variational Gaussian Process for Stochastic Optimisation with Flux.jl](@ref). For an example of non-conjugate inference, see [Classification: Sparse Variational Approximation for Non-Conjugate Likelihoods with Optim's L-BFGS](@ref).

# Available Parametrizations

Two parametrizations of `q(u)` are presently available: [`Centered`](@ref) and [`NonCentered`](@ref).
The `Centered` parametrization expresses `q(u)` directly in terms of its mean and covariance.
The `NonCentered` parametrization instead parametrizes the mean and covariance of
`ε := cholesky(cov(u)).U' \ (u - mean(u))`.
These parametrizations are also known respectively as "Unwhitened" and "Whitened".

The choice of parametrization can have a substantial impact on the time it takes for ELBO
optimization to converge, and which parametrization is better in a particular situation is
not generally obvious.
That being said, the `NonCentered` parametrization often converges in fewer iterations, so it is the default --
it is what is used in all of the examples above.

If you require a particular parametrization, simply use the 3-argument version of the
approximation constructor:
```julia
SparseVariationalApproximation(Centered(), fz, q)
SparseVariationalApproximation(NonCentered(), fz, q)
```

For a general discussion around these two parametrizations, see e.g. [^Gorinova].
For a GP-specific discussion, see e.g. section 3.4 of [^Paciorek].

[^Gorinova]: Gorinova, Maria and Moore, Dave and Hoffman, Matthew [Automatic Reparameterisation of Probabilistic Programs](http://proceedings.mlr.press/v119/gorinova20a)
[^Paciorek]: [Paciorek, Christopher Joseph. Nonstationary Gaussian processes for regression and spatial modelling. Diss. Carnegie Mellon University, 2003.](https://www.stat.berkeley.edu/~paciorek/diss/paciorek-thesis.pdf)
