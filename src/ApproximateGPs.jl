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

using AbstractGPs: AbstractGP, FiniteGP, LatentFiniteGP, ApproxPosteriorGP, At_A, diag_At_A

export SVGP, DefaultQuadrature, Analytic, GaussHermite, MonteCarlo

include("utils.jl")
include("svgp.jl")
include("expected_loglik.jl")
include("elbo.jl")

using ForwardDiff

export LaplaceApproximation
export laplace_lml, build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
export laplace_steps  # TODO clean up/discard
include("laplace.jl")

end
