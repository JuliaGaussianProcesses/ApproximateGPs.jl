module NearestNeighborsModule
using ..API
using ChainRulesCore
using KernelFunctions, LinearAlgebra, SparseArrays, AbstractGPs

@doc raw"""
Constructs the matrix $B$ for which $f = Bf + \epsilon$ were $f$
are the values of the GP and $\epsilon$ is a vector of zero mean
independent Gaussian noise. 
This matrix builds the conditional mean for function value $f_i$
in terms of the function values for previous $f_j$, where $j < i$.
See equation (9) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
function make_B(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
    n = length(pts)
    len = ((k + 1) * k) ÷ 2 + (n - k - 1) * k
    js = Int[]
    is = Int[]
    vals = T[]
    sizehint!(js, len)
    sizehint!(is, len)
    sizehint!(vals, len)
    for i in 2:n
        ns = pts[max(1, i-k):i-1]
        row = get_row(kern, ns, pts[i])
        start_ix = max(i-k, 1)
        col_ixs = start_ix:(start_ix + length(row) - 1)
        append!(js, col_ixs)
        append!(is, fill(i, length(col_ixs)))
        append!(vals, row)
    end
    return sparse(is, js, vals, n, n)
end

@doc raw"""
Constructs the nonzero entries of a row in the matrix $B$
for which $f = Bf + \epsilon$ for Gaussian process values $f$.
""" 
function get_row(kern, ns, p)
    return kernelmatrix(kern,ns) \ kern.(ns, p)
end

function ChainRulesCore.rrule(cfg::RuleConfig, ::typeof(make_B), pts::AbstractVector{T}, k, kern) where {T}
    n = length(pts)
    js = Array{Vector{Int}}(undef, n - 1)
    is = Array{Vector{Int}}(undef, n - 1)
    vals = Array{Vector{T}}(undef, n - 1)
    pbs = Array{Function}(undef, n -1)
    for i in 2:n
        ns = pts[max(1, i-k):i-1]
        row, pb = rrule_via_ad(cfg, get_row, kern, ns, pts[i])
        start_ix = max(i-k, 1)
        col_ixs = start_ix:(start_ix + length(row) - 1)
        js[i-1] = col_ixs
        is[i-1] = fill(i, length(col_ixs))
        vals[i-1] = row
        pbs[i-1] = pb
    end   
    function pullback(Δy)
      d_kern = sum(pbs[i](Δy[is[i][1], js[i]])[1] for i in 1:length(is))
      (NoTangent(), NoTangent(), NoTangent(), d_kern)
    end
    return sparse(reduce(vcat, is), reduce(vcat, js),
        reduce(vcat, vals), n, n), pullback       
end
    
@doc raw"""
Constructs the diagonal covariance matrix for noise vector $\epsilon$
for which $f = Bf + \epsilon$. 
See equation (10) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
function make_F(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
    n = length(pts)
    vals = [
        begin
            prior = kern(pts[i], pts[i])
            if i == 1
                prior
            else
                ns = pts[max(1, i-k):i-1]
                ki = kern.(ns, pts[i])
                prior - dot(ki, kernelmatrix(kern, ns) \ ki)
            end
        end
    for i in 1:n]
    return Diagonal(vals)
end

struct NearestNeighbors
    k::Int
end

"`InvRoot(U)` is a lazy representation of `inv(UU')`"
struct InvRoot{A}
    U::A
end

LinearAlgebra.logdet(A::InvRoot) = -2 * logdet(A.U) 

AbstractGPs.diag_Xt_invA_X(A::InvRoot, X::AbstractVecOrMat) = AbstractGPs.diag_At_A(A.U' * X)

AbstractGPs.Xt_invA_X(A::InvRoot, X::AbstractVecOrMat) = AbstractGPs.At_A(A.U' * X)

"""
Make a sparse approximation of the square root of the precision matrix
"""
function approx_root_prec(x::AbstractVector, k::Int, kern::Kernel)
    F = make_F(x, k, kern)
    B = make_B(x, k, kern)
    UpperTriangular((I - B)' * inv(sqrt(F)))
end

function AbstractGPs.posterior(nn::NearestNeighbors, fx::AbstractGPs.FiniteGP, y::AbstractVector)
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
  return -0.5 * ld -(length(y)/2) * log(2 * pi) - 0.5 * quadform
end

end
