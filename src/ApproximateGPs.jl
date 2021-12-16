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
import ChainRulesCore
using FillArrays: Fill
using KLDivergences: KL
using IrrationalConstants: log2π

using AbstractGPs: AbstractGP, FiniteGP, LatentFiniteGP, ApproxPosteriorGP, At_A, diag_At_A

export SVGP, DefaultQuadrature, Analytic, GaussHermite, MonteCarlo

include("utils.jl")
include("svgp.jl")
include("expected_loglik.jl")
include("elbo.jl")

import ForwardDiff

export LaplaceApproximation
export build_laplace_objective, build_laplace_objective!
export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221
include("laplace.jl")

using Random: randperm

export ExpectationPropagation
include("ep.jl")

end
