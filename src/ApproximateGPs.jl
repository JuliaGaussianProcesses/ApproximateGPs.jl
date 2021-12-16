module ApproximateGPs

using Reexport
@reexport using AbstractGPs
@reexport using GPLikelihoods
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using FastGaussQuadrature: gausshermite
#using SpecialFunctions
using ChainRulesCore: ignore_derivatives, NoTangent
using ChainRulesCore: ChainRulesCore
using FillArrays: Fill
using KLDivergences: KL
using IrrationalConstants: log2π

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

export SparseVariationalApproximation, Centered, NonCentered
export DefaultQuadrature, Analytic, GaussHermite, MonteCarlo

include("utils.jl")
include("expected_loglik.jl")
include("sparse_variational.jl")

using ForwardDiff: ForwardDiff

export LaplaceApproximation
export build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
include("laplace.jl")

using Random: randperm

export ExpectationPropagation
include("ep.jl")

include("deprecations.jl")

end
