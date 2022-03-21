module ApproximateGPs

using Reexport

@reexport using GPLikelihoods

include("API.jl")
@reexport using .API: approx_lml

include("utils.jl")

include("SparseVariationalApproximationModule.jl")
@reexport using .SparseVariationalApproximationModule:
    SparseVariationalApproximation, Centered, NonCentered
@reexport using .SparseVariationalApproximationModule:
    DefaultQuadrature, Analytic, GaussHermite, MonteCarlo
@reexport using .SparseVariationalApproximationModule:
    PseudoObsSparseVariationalApproximation,
    ObsCovLikelihood,
    DecoupledObsCovLikelihood

include("LaplaceApproximationModule.jl")
@reexport using .LaplaceApproximationModule: LaplaceApproximation
@reexport using .LaplaceApproximationModule:
    build_laplace_objective, build_laplace_objective!

include("deprecations.jl")

end
