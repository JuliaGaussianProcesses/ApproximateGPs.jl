using Random
using Test
using SparseGPs

const GROUP = get(ENV, "GROUP", "All")
const PKGDIR = dirname(dirname(pathof(SparseGPs)))

@testset "SparseGPs" begin
    include("svgp.jl")
    println(" ")
    @info "Ran svgp tests"
end
