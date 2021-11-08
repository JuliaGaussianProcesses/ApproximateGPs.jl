# Parametrisations.
struct Centred end
struct NonCentred end

struct SparseVariationalApproximation{Parametrisation,Tfz<:FiniteGP,Tq<:AbstractMvNormal}
    fz::Tfz
    q::Tq
end

"""
    SparseVariationalApproximation(::Parametrisation, fz::FiniteGP, q::AbstractMvNormal)

Packages the prior over the pseudo-points, `fz`, and the approximate posterior at the
pseudo-points, `q`, together into a single object.
"""
function SparseVariationalApproximation(
    ::Parametrisation, fz::Tfz, q::Tq
) where {Parametrisation,Tfz<:FiniteGP,Tq<:AbstractMvNormal}
    return SparseVariationalApproximation{Parametrisation,Tfz,Tq}(fz, q)
end

"""
    SparseVariationalApproximation(fz::FiniteGP, q::AbstractMvNormal)

Packages the prior over the pseudo-points `fz`, and the approximate posterior at the
pseudo-points, which is `mean(fz) + cholesky(cov(fz)).U' * ε`, `ε ∼ q`.

Shorthand for
```julia
SparseVariationalApproximation(NonCentred(), fz, q)
```
"""
function SparseVariationalApproximation(fz::FiniteGP, q::AbstractMvNormal)
    return SparseVariationalApproximation(NonCentred(), fz, q)
end

raw"""
    posterior(sva::SparseVariationalApproximation{Centred})

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
function AbstractGPs.posterior(sva::SparseVariationalApproximation{Centred})
    q, fz = sva.q, sva.fz
    m, S = mean(q), _chol_cov(q)
    Kuu = _chol_cov(fz)
    B = Kuu.L \ S.L
    α = Kuu \ (m - mean(fz))
    data = (S=S, m=m, Kuu=Kuu, B=B, α=α)
    return ApproxPosteriorGP(sva, fz.f, data)
end

function AbstractGPs.posterior(
    sva::SparseVariationalApproximation, fx::FiniteGP, ::AbstractVector{<:Real}
)
    @assert sva.fz.f === fx.f
    return posterior(sva)
end

#
# Code below this point just implements the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

function Statistics.mean(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}}, x::AbstractVector
)
    return mean(f.prior, x) + cov(f.prior, x, inducing_points(f)) * f.data.α
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
end

function Statistics.var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}},
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
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ = cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
    return μ, Σ
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centred}}, x::AbstractVector
)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ_diag = var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
    return μ, Σ_diag
end

#
# NonCentred parametrisation.
#

raw"""
    posterior(sva::SparseVariationalApproximation{NonCentred})

Compute the approximate posterior [1] over the process `f =
sva.fz.f`, given inducing inputs `z = sva.fz.x` and a variational
distribution over inducing points `sva.q` (which represents ``q(ε)``
where `ε = cholesky(cov(fz)).U' \ (f(z) - mean(f(z)))`). The approximate posterior at test
points ``x^*`` where ``f^* = f(x^*)`` is then given by:

```math
q(f^*) = \int p(f | ε) q(ε) du
```
which can be found in closed form.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.posterior(approx::SparseVariationalApproximation{NonCentred})
    fz = approx.fz
    data = (Cuu=_chol_cov(fz), C_ε=_chol_cov(approx.q))
    return ApproxPosteriorGP(approx, fz.f, data)
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
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}}, x::AbstractVector
)
    return mean(f.prior, x) + _A(f, x)' * mean(f.approx.q)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}}, x::AbstractVector
)
    A = _A(f, x)
    return cov(f.prior, x) - At_A(A) + Xt_A_X(f.data.C_ε, A)
end

function Statistics.var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}}, x::AbstractVector
)
    A = _A(f, x)
    return var(f.prior, x) - diag_At_A(A) + diag_Xt_A_X(f.data.C_ε, A)
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}},
    x::AbstractVector,
    y::AbstractVector,
)
    Ax = _A(f, x)
    Ay = _A(f, y)
    return cov(f.prior, x, y) - Ax'Ay + Xt_A_Y(Ax, f.data.C_ε, Ay)
end

function StatsBase.mean_and_cov(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}}, x::AbstractVector
)
    A = _A(f, x)
    μ = mean(f.prior, x) + A' * mean(f.approx.q)
    Σ = cov(f.prior, x) - At_A(A) + Xt_A_X(f.data.C_ε, A)
    return μ, Σ
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentred}}, x::AbstractVector
)
    A = _A(f, x)
    μ = mean(f.prior, x) + A' * mean(f.approx.q)
    Σ = var(f.prior, x) - diag_At_A(A) + diag_Xt_A_X(f.data.C_ε, A)
    return μ, Σ
end

#
# Misc utility.
#

inducing_points(f::ApproxPosteriorGP{<:SparseVariationalApproximation}) = f.approx.fz.x

_chol_cov(q::AbstractMvNormal) = cholesky(Symmetric(cov(q)))
_chol_cov(q::MvNormal) = cholesky(q.Σ)
