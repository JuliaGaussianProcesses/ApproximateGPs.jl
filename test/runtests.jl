using Random
using Test
using ApproximateGPs
using Flux
using IterTools
using AbstractGPs
using Distributions
using LinearAlgebra
using PDMats

const GROUP = get(ENV, "GROUP", "All")
const PKGDIR = dirname(dirname(pathof(ApproximateGPs)))

include("test_utils.jl")

@testset "ApproximateGPs" begin
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
