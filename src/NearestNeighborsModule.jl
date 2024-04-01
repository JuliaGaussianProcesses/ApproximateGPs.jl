module NearestNeighborsModule
using ..API
using ChainRulesCore
using KernelFunctions, LinearAlgebra, SparseArrays, AbstractGPs, IrrationalConstants

"""
Constructs the matrix ``B`` for which ``f = Bf + \epsilon`` where ``f``
are the values of the GP and ``\epsilon`` is a vector of zero mean
independent Gaussian noise. 
This matrix builds the conditional mean for function value ``f_i``
in terms of the function values for previous ``f_j``, where ``j < i``.
See equation (9) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
function make_B(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
    rows = make_rows(pts, k, kern)
    js = make_js(rows, k)
    is = make_is(js)
    n = length(pts)
    return sparse(reduce(vcat, is), reduce(vcat, js), reduce(vcat, rows), n, n)
end

function make_rows(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
    return [make_row(kern, pts[max(1, i - k):(i - 1)], pts[i]) for i in 2:length(pts)]
end

function make_row(kern::Kernel, ns::AbstractVector{T}, p::T) where {T}
    return kernelmatrix(kern, ns) \ kern.(ns, p)
end

function make_js(rows, k)
    return map(zip(rows, 2:(length(rows) + 1))) do (row, i)
        start_ix = max(i - k, 1)
        return start_ix:(start_ix + length(row) - 1)
    end
end

make_is(js) = [fill(i, length(col_ix)) for (col_ix, i) in zip(js, 2:(length(js) + 1))]

"""
Constructs the diagonal covariance matrix for noise vector ``\epsilon``
for which ``f = Bf + \epsilon``. 
See equation (10) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
function make_F(pts::AbstractVector, k::Int, kern::Kernel)
    n = length(pts)
    vals = [
        begin
            prior = kern(pts[i], pts[i])
            if i == 1
                prior
            else
                ns = pts[max(1, i - k):(i - 1)]
                ki = kern.(ns, pts[i])
                prior - dot(ki, kernelmatrix(kern, ns) \ ki)
            end
        end for i in 1:n
    ]
    return Diagonal(vals)
end

@doc raw"""
In a ``k``-nearest neighbor (or Vecchia) Gaussian Process approximation,
we assume that the joint distribution ``p(f_1, f_2, f_3, \dotsc)``
factors as ``\prod_i p(f_i | f_{i-1}, \dotsc f_{i-k})``, where each ``f_i``
is only influenced by its ``k`` previous neighbors. This allows us to express
the vector ``f`` as ``Bf + \epsilon`` where ``B`` is a sparse matrix with only
``k`` entries per row and ``\epsilon`` is Gaussian distributed with diagonal
covariance ``F``. The precision matrix of the Gaussian process at the
specified points simplifies to ``(I-B)'F^{-1}(I-B)``. 
"""
struct NearestNeighbors
    k::Int
end

"`InvRoot(U)` is a lazy representation of `inv(UU')`"
struct InvRoot{A}
    U::A
end

LinearAlgebra.logdet(A::InvRoot) = -2 * logdet(A.U)

function AbstractGPs.diag_Xt_invA_X(A::InvRoot, X::AbstractVecOrMat)
    return AbstractGPs.diag_At_A(A.U' * X)
end

AbstractGPs.Xt_invA_X(A::InvRoot, X::AbstractVecOrMat) = AbstractGPs.At_A(A.U' * X)

# Make a sparse approximation of the square root of the precision matrix
function approx_root_prec(x::AbstractVector, k::Int, kern::Kernel)
    F = make_F(x, k, kern)
    B = make_B(x, k, kern)
    return UpperTriangular((I - B)' * inv(sqrt(F)))
end

function AbstractGPs.posterior(
    nn::NearestNeighbors, fx::AbstractGPs.FiniteGP, y::AbstractVector
)
    kern = fx.f.kernel
    U = approx_root_prec(fx.x, nn.k, kern)
    δ = y - mean(fx)
    α = U * (U' * δ)
    C = InvRoot(U)
    return AbstractGPs.PosteriorGP(fx.f, (α=α, C=C, x=fx.x, δ=δ))
end

function API.approx_lml(nn::NearestNeighbors, fx::AbstractGPs.FiniteGP, y::AbstractVector)
    post = posterior(nn, fx, y)
    quadform = post.data.α' * post.data.δ
    ld = logdet(post.data.C)
    return -(ld + length(y) * eltype(y)(log2π) + quadform) / 2
end

end
