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

using AbstractGPs:
    FiniteGP,
    LatentFiniteGP,
    ApproxPosteriorGP,
    _cholesky,
    _symmetric,
    At_A,
    diag_At_A,
    Xt_invA_X

export elbo,
    approx_posterior,
    SVGP

include("elbo.jl")
include("svgp.jl")

end
