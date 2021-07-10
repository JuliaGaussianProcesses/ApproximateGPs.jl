module SparseGPs

using AbstractGPs
using Distributions
using Optim
using StatsFuns
using LinearAlgebra
using Statistics
using StatsBase
using FastGaussQuadrature
using GPLikelihoods
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
