using Random
using Test
using ApproximateGPs
using Flux
using IterTools
using AbstractGPs
using AbstractGPs: LatentFiniteGP
using Distributions
using LogExpFunctions: logistic
using LinearAlgebra
using PDMats
using Optim
using Zygote
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using QuadGK

const GROUP = get(ENV, "GROUP", "All")
const PKGDIR = dirname(dirname(pathof(ApproximateGPs)))

include("test_utils.jl")

@testset "ApproximateGPs" begin
    include("expected_loglik.jl")
    println(" ")
    @info "Ran expected_loglik tests"
    @testset "SVGP" begin
        include("svgp.jl")
        println(" ")
        @info "Ran svgp tests"
        include("elbo.jl")
        println(" ")
        @info "Ran elbo tests"
        include("equivalences.jl")
        println(" ")
        @info "Ran equivalences tests"
    end

    include("common_test_support.jl")

    @testset "Laplace" begin
        include("laplace.jl")
        println(" ")
        @info "Ran laplace tests"
    end

    @testset "EP" begin
        include("ep.jl")
        println(" ")
        @info "Ran ep tests"
    end
end
