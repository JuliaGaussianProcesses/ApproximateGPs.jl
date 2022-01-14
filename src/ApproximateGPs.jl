module ApproximateGPs

using Reexport

include("API.jl")
@reexport using .API

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

include("SparseVariationalApproximationModule.jl")
@reexport using .SparseVariationalApproximationModule

include("LaplaceApproximationModule.jl")
@reexport using .LaplaceApproximationModule

include("deprecations.jl")

end
