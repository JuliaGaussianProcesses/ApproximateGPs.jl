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
using PDMats: chol_lower

using AbstractGPs: AbstractGP, FiniteGP, LatentFiniteGP, ApproxPosteriorGP, At_A, diag_At_A

include("utils.jl")

export DefaultQuadrature, Analytic, GaussHermite, MonteCarlo
include("expected_loglik.jl")

export SparseVariationalApproximation, Centered, NonCentered
include("sparse_variational.jl")

using ForwardDiff

export LaplaceApproximation
export build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
include("laplace.jl")

include("deprecations.jl")

end
