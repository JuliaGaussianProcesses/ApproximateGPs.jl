module ApproximateGPs

using Reexport

include("API.jl")
@reexport using .API

include("utils.jl")

include("SparseVariationalApproximationModule.jl")
@reexport using .SparseVariationalApproximationModule

include("LaplaceApproximationModule.jl")
@reexport using .LaplaceApproximationModule

include("deprecations.jl")

end
