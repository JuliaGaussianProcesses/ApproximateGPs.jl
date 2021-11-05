"""
    WhitenedSparseVariationalApproximation(fz::FiniteGP, q_ε::AbstractMvNormal)

Packages the prior over the pseudo-points `fz`, and the approximate posterior at the
pseudo-points, which is `mean(fz) + cholesky(cov(fz)).U' * ε`, `ε ∼ q_ε`.
"""
struct WhitenedSparseVariationalApproximation{Tfz<:FiniteGP,Tq_ε<:AbstractMvNormal} <:
       AbstractSparseVariationalApproximation
    fz::Tfz
    q_ε::Tq_ε
end

raw"""
    posterior(sva::WhitenedSparseVariationalApproximation)

Compute the approximate posterior [1] over the process `f =
sva.fz.f`, given inducing inputs `z = sva.fz.x` and a variational
distribution over inducing points `sva.q_ε` (which represents ``q(ε)``
where `ε = cholesky(cov(fz)).U' * (f(z) - mean(f(z)))`). The approximate posterior at test
points ``x^*`` where ``f^* = f(x^*)`` is then given by:

```math
q(f^*) = \int p(f | ε) q(ε) du
```
which can be found in closed form.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.posterior(approx::WhitenedSparseVariationalApproximation)
    fz = approx.fz
    data = (Cuu=_chol_cov(fz), C_ε=_chol_cov(approx.q_ε))
    return ApproxPosteriorGP(approx, fz.f, data)
end

function AbstractGPs.posterior(
    approx::WhitenedSparseVariationalApproximation, fx::FiniteGP, ::AbstractVector
)
    @assert approx.fz.f === fx.f
    return posterior(approx)
end

#
# Code below this point just implements the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

# Produces a matrix that is consistently referred to as A in this file. A more descriptive
# name is, unfortunately, not obvious. It's just an intermediate quantity that happens to
# get used a lot.
_A(f, x) = f.data.Cuu.U' \ cov(f.prior, inducing_points(f), x)

function Statistics.mean(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation}, x::AbstractVector
)
    return mean(f.prior, x) + _A(f, x)' * mean(f.approx.q_ε)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    return cov(f.prior, x) - At_A(A) + Xt_A_X(f.data.C_ε, A)
end

function Statistics.var(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    return var(f.prior, x) - diag_At_A(A) + diag_Xt_A_X(f.data.C_ε, A)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation},
    x::AbstractVector,
    y::AbstractVector,
)
    Ax = _A(f, x)
    Ay = _A(f, y)
    return cov(f.prior, x, y) - Ax'Ay + Xt_A_Y(Ax, f.data.C_ε, Ay)
end

function StatsBase.mean_and_cov(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    μ = mean(f.prior, x) + A' * mean(f.approx.q_ε)
    Σ = cov(f.prior, x) - At_A(A) + Xt_A_X(f.data.C_ε, A)
    return μ, Σ
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    μ = mean(f.prior, x) + A' * mean(f.approx.q_ε)
    Σ = var(f.prior, x) - diag_At_A(A) + diag_Xt_A_X(f.data.C_ε, A)
    return μ, Σ
end

function inducing_points(f::ApproxPosteriorGP{<:WhitenedSparseVariationalApproximation})
    return f.approx.fz.x
end
