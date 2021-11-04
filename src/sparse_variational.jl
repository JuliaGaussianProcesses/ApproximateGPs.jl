struct SparseVariationalApproximation{Tfz<:FiniteGP,Tq<:AbstractMvNormal}
    fz::Tfz
    q::Tq
end

raw"""
    posterior(sva::SparseVariationalApproximation)

Compute the approximate posterior [1] over the process `f =
sva.fz.f`, given inducing inputs `z = sva.fz.x` and a variational
distribution over inducing points `sva.q` (which represents ``q(u)``
where `u = f(z)`). The approximate posterior at test points ``x^*``
where ``f^* = f(x^*)`` is then given by:

```math
q(f^*) = \int p(f | u) q(u) du
```
which can be found in closed form.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.posterior(sva::SparseVariationalApproximation)
    q, fz = sva.q, sva.fz
    m, S = mean(q), _chol_cov(q)
    Kuu = _chol_cov(fz)
    B = Kuu.L \ S.L
    α = Kuu \ (m - mean(fz))
    data = (S=S, m=m, Kuu=Kuu, B=B, α=α)
    return ApproxPosteriorGP(sva, fz.f, data)
end

function AbstractGPs.posterior(
    sva::SparseVariationalApproximation, fx::FiniteGP, ::AbstractVector
)
    @assert sva.fz.f === fx.f
    return posterior(sva)
end

function Statistics.mean(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation}, x::AbstractVector
)
    return mean(f.prior, x) + cov(f.prior, x, inducing_points(f)) * f.data.α
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
end

function Statistics.var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation},
    x::AbstractVector,
    y::AbstractVector,
)
    B = f.data.B
    Cxu = cov(f.prior, x, inducing_points(f))
    Cuy = cov(f.prior, inducing_points(f), y)
    D = f.data.Kuu.L \ Cuy
    E = Cxu / f.data.Kuu.L'
    return cov(f.prior, x, y) - (E * D) + (E * B * B' * D)
end

function StatsBase.mean_and_cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ = cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
    return μ, Σ
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ_diag = var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
    return μ, Σ_diag
end

inducing_points(f::ApproxPosteriorGP{<:SparseVariationalApproximation}) = f.approx.fz.x

_chol_cov(q::AbstractMvNormal) = cholesky(Symmetric(cov(q)))
_chol_cov(q::MvNormal) = cholesky(q.Σ)
