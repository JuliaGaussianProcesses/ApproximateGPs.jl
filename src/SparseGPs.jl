module SparseGPs

using AbstractGPs
using Distributions
using Optim
using StatsFuns
using LinearAlgebra
using Statistics
using StatsBase

using AbstractGPs: FiniteGP, ApproxPosteriorGP, _cholesky, _symmetric, Xt_invA_X, diag_At_A

export elbo,
    approx_posterior,
    SVGP

include("svgp.jl")

end
