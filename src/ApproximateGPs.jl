module ApproximateGPs

using Reexport
@reexport using AbstractGPs
@reexport using GPLikelihoods
using Distributions
using LinearAlgebra
using Random
using Statistics
using StatsBase
using FastGaussQuadrature
using SpecialFunctions
using ChainRulesCore
using FillArrays
using PDMats: chol_lower
using IrrationalConstants: sqrt2, invsqrtπ

using AbstractGPs:
    AbstractGP,
    FiniteGP,
    LatentFiniteGP,
    ApproxPosteriorGP,
    inducing_points,
    At_A,
    diag_At_A,
    Xt_A_X,
    Xt_invA_X

include("utils.jl")

export DefaultQuadrature, Analytic, GaussHermite, MonteCarlo
include("expected_loglik.jl")

export SparseVariationalApproximation, Centered, NonCentered
include("sparse_variational.jl")
include("pathwise_sampling.jl")

using ForwardDiff

export LaplaceApproximation
export build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
include("laplace.jl")

include("deprecations.jl")

end
