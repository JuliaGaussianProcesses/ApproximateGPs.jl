module ApproximateGPs

using Reexport

@reexport using AbstractGPs
@reexport using GPLikelihoods

include("API.jl")
@reexport using .API: approx_lml

include("utils.jl")

include("SparseVariationalApproximationModule.jl")
@reexport using .SparseVariationalApproximationModule:
    SparseVariationalApproximation, Centered, NonCentered

include("LaplaceApproximationModule.jl")
@reexport using .LaplaceApproximationModule: LaplaceApproximation
@reexport using .LaplaceApproximationModule:
    build_laplace_objective, build_laplace_objective!

include("NearestNeighborsModule.jl")
@reexport using .NearestNeighborsModule: NearestNeighbors

include("deprecations.jl")

include("TestUtils.jl")

import ChainRulesCore: ProjectTo, Tangent
using PDMats: ScalMat
ProjectTo(x::T) where {T<:ScalMat} = ProjectTo{T}(; dim=x.dim, value=ProjectTo(x.value))
(pr::ProjectTo{<:ScalMat})(dx::ScalMat) = ScalMat(pr.dim, pr.value(dx.value))
(pr::ProjectTo{<:ScalMat})(dx::Tangent{<:ScalMat}) = ScalMat(pr.dim, pr.value(dx.value))

end
