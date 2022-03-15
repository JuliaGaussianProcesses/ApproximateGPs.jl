# ApproximateGPs

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/ApproximateGPs.jl/dev)
[![CI](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/ApproximateGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/ApproximateGPs.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## Aim of this package

Provide various algorithms for approximate inference in latent Gaussian process models, currently focussing on non-conjugate (non-Gaussian) likelihoods and sparse approximations.

## Structure

Each approximation lives in its own submodule (`<Approximation>Module`), though
in general using the exported API is sufficient.

The main API is:

* `posterior(approximation, lfx::LatentFiniteGP, ys)` to obtain the posterior
  approximation to `lfx` conditioned on the observations `ys`.

* `approx_lml(approximation, lfx::LatentFiniteGP, ys)` which returns the
  marginal likelihood approximation that can be used for hyperparameter
  optimisation.

Currently implemented approximations:

* `LaplaceApproximation`

* `SparseVariationalApproximation`

  NOTE: requires optimisation of the variational distribution even for fixed
  hyperparameters.
