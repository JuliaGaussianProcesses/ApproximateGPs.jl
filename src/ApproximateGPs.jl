module ApproximateGPs

using Reexport
@reexport using AbstractGPs
@reexport using GPLikelihoods
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using FastGaussQuadrature
using SpecialFunctions
using ChainRulesCore
using FillArrays
using KLDivergences

using AbstractGPs:
    AbstractGP,
    FiniteGP,
    LatentFiniteGP,
    ApproxPosteriorGP,
    At_A,
    diag_At_A,
    Xt_A_X,
    Xt_A_Y,
    diag_Xt_A_X

export SparseVariationalApproximation, WhitenedSparseVariationalApproximation
export DefaultQuadrature, Analytic, GaussHermite, MonteCarlo

include("utils.jl")
include("sparse_variational.jl")
include("whitened_sparse_variational.jl")
include("expected_loglik.jl")
include("elbo.jl")

using ForwardDiff

export LaplaceApproximation
export build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
include("laplace.jl")

include("deprecations.jl")

end
