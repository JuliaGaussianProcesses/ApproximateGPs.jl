# These methods to create a Cholesky directly from the factorisation will be in Julia 1.7
# https://github.com/JuliaLang/julia/pull/39352
if VERSION < v"1.7"
    LinearAlgebra.Cholesky(L::LowerTriangular{T}) where {T} = Cholesky{T,typeof(L.data)}(L.data, 'L', 0)
    LinearAlgebra.Cholesky(U::UpperTriangular{T}) where {T} = Cholesky{T,typeof(U.data)}(U.data, 'U', 0)
end
