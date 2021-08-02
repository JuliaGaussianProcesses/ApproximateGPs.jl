module SparseGPs

using AbstractGPs
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using FastGaussQuadrature
using GPLikelihoods
using SpecialFunctions
using ChainRulesCore
using PDMats
using Functors
using FillArrays
using KLDivergences


using AbstractGPs:
    AbstractGP, FiniteGP, LatentFiniteGP, ApproxPosteriorGP, At_A, diag_At_A

export SVGP, SVGPModel, Default, Analytic, Quadrature, MonteCarlo, prior, loss

include("utils.jl")
include("elbo.jl")
include("svgp.jl")
include("models.jl")

end
