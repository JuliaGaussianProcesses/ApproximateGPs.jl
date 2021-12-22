# These methods to create a Cholesky directly from the factorisation will be in Julia 1.7
# https://github.com/JuliaLang/julia/pull/39352
if VERSION < v"1.7"
    function LinearAlgebra.Cholesky(L::LowerTriangular{T}) where {T}
        return Cholesky{T,typeof(L.data)}(L.data, 'L', 0)
    end
    function LinearAlgebra.Cholesky(U::UpperTriangular{T}) where {T}
        return Cholesky{T,typeof(U.data)}(U.data, 'U', 0)
    end
end

_chol_cov(q::AbstractMvNormal) = cholesky(Symmetric(cov(q)))
_chol_cov(q::MvNormal) = cholesky(q.Σ)

_cov(q::MvNormal) = q.Σ
