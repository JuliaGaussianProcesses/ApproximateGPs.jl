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
using QuadGK  # TODO replace with FastGaussQuadrature

export laplace_lml, build_laplace_objective, build_laplace_objective!
export laplace_posterior
export laplace_steps
include("laplace.jl")
include("ep.jl")

end
