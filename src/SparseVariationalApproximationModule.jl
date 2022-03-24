module SparseVariationalApproximationModule

using ..API

export SparseVariationalApproximation,
    Centered, NonCentered, PseudoObsSparseVariationalApproximation

using ..ApproximateGPs: _chol_cov, _cov
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using FillArrays: Fill
using PDMats: chol_lower
using IrrationalConstants: sqrt2, invsqrtπ

using AbstractGPs: AbstractGPs
using AbstractGPs:
    AbstractGP,
    FiniteGP,
    LatentFiniteGP,
    ApproxPosteriorGP,
    elbo,
    posterior,
    marginals,
    At_A,
    diag_At_A
using GPLikelihoods: GaussianLikelihood

export DefaultQuadrature, Analytic, GaussHermite, MonteCarlo
include("expected_loglik.jl")

"""
    abstract type AbstractSparseVariationalApproximation end

Supertype for sparse variational approximations.
"""
abstract type AbstractSparseVariationalApproximation end

@doc raw"""
    Centered()

Used in conjunction with `SparseVariationalApproximation`.
States that the `q` field of [`SparseVariationalApproximation`](@ref) is to be interpreted
directly as the approximate posterior over the pseudo-points.

This is also known as the "unwhitened" parametrization [1].

See also [`NonCentered`](@ref).

[1] - https://en.wikipedia.org/wiki/Whitening_transformation
"""
struct Centered end

@doc raw"""
    NonCentered()

Used in conjunction with `SparseVariationalApproximation`.
States that the `q` field of [`SparseVariationalApproximation`](@ref) is to be interpreted
as the approximate posterior over `cholesky(cov(u)).L \ (u - mean(u))`, where `u` are the
pseudo-points.

This is also known as the "whitened" parametrization [1].

See also [`Centered`](@ref).

[1] - https://en.wikipedia.org/wiki/Whitening_transformation
"""
struct NonCentered end

struct SparseVariationalApproximation{
    Parametrization,Tfz<:FiniteGP,Tq<:AbstractMvNormal
} <: AbstractSparseVariationalApproximation
    fz::Tfz
    q::Tq
end

@doc raw"""
    SparseVariationalApproximation(
        ::Parametrization, fz::FiniteGP, q::AbstractMvNormal
    ) where {Parametrization}

Produce a `SparseVariationalApproximation{Parametrization}`, which packages the prior over
the pseudo-points, `fz`, and the approximate posterior at the pseudo-points, `q`, together
into a single object.

The `Parametrization` determines the precise manner in which `q` and `fz` are interpreted.
Existing parametrizations include [`Centered`](@ref) and [`NonCentered`](@ref).
"""
function SparseVariationalApproximation(
    ::Parametrization, fz::Tfz, q::Tq
) where {Parametrization,Tfz<:FiniteGP,Tq<:AbstractMvNormal}
    return SparseVariationalApproximation{Parametrization,Tfz,Tq}(fz, q)
end

"""
    SparseVariationalApproximation(fz::FiniteGP, q::AbstractMvNormal)

Packages the prior over the pseudo-points `fz`, and the approximate posterior at the
pseudo-points, which is `mean(fz) + cholesky(cov(fz)).L * ε`, `ε ∼ q`.

Shorthand for
```julia
SparseVariationalApproximation(NonCentered(), fz, q)
```
"""
function SparseVariationalApproximation(fz::FiniteGP, q::AbstractMvNormal)
    return SparseVariationalApproximation(NonCentered(), fz, q)
end

@doc raw"""
    posterior(sva::SparseVariationalApproximation{Centered})

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
function AbstractGPs.posterior(sva::SparseVariationalApproximation{Centered})
    # m* = K*u Kuu⁻¹ (mean(q) - mean(fz))
    #    = K*u α
    # Centered: α = Kuu⁻¹ (m - mean(fz))
    # [NonCentered: α = Lk⁻ᵀ m]
    # V** = K** - K*u (Kuu⁻¹ - Kuu⁻¹ cov(q) Kuu⁻¹) Ku*
    #     = K** - K*u (Kuu⁻¹ - Kuu⁻¹ cov(q) Kuu⁻¹) Ku*
    #     = K** - (K*u Lk⁻ᵀ) (Lk⁻¹ Ku*) + (K*u Lk⁻ᵀ) Lk⁻¹ cov(q) Lk⁻ᵀ (Lk⁻¹ Ku*)
    #     = K** - A'A + A' Lk⁻¹ cov(q) Lk⁻ᵀ A
    #     = K** - A'A + A' Lk⁻¹ Lq Lqᵀ Lk⁻ᵀ A
    #     = K** - A'A + A' B B' A
    # A = Lk⁻¹ Ku*
    # Centered: B = Lk⁻¹ Lq
    # [NonCentered: B = Lq]
    q, fz = sva.q, sva.fz
    m, S = mean(q), _chol_cov(q)
    Kuu = _chol_cov(fz)
    B = chol_lower(Kuu) \ chol_lower(S)
    α = Kuu \ (m - mean(fz))
    data = (Kuu=Kuu, B=B, α=α)
    return ApproxPosteriorGP(sva, fz.f, data)
end

#
# NonCentered Parametrization.
#

@doc raw"""
    posterior(sva::SparseVariationalApproximation{NonCentered})

Compute the approximate posterior [1] over the process `f =
sva.fz.f`, given inducing inputs `z = sva.fz.x` and a variational
distribution over inducing points `sva.q` (which represents ``q(ε)``
where `ε = cholesky(cov(fz)).L \ (f(z) - mean(f(z)))`). The approximate posterior at test
points ``x^*`` where ``f^* = f(x^*)`` is then given by:

```math
q(f^*) = \int p(f | ε) q(ε) du
```
which can be found in closed form.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.posterior(sva::SparseVariationalApproximation{NonCentered})
    # u = Lk v + mean(fz), v ~ q
    # m* = K*u Kuu⁻¹ Lk (mean(u) - mean(fz))
    #    = K*u (Lk Lkᵀ)⁻¹ Lk mean(q)
    #    = K*u Lk⁻ᵀ Lk⁻¹ Lk mean(q)
    #    = K*u Lk⁻ᵀ mean(q)
    #    = K*u α
    # NonCentered: α = Lk⁻ᵀ m
    # [Centered: α = Kuu⁻¹ (m - mean(fz))]
    # V** = K** - K*u (Kuu⁻¹ - Kuu⁻¹ Lk cov(q) Lkᵀ Kuu⁻¹) Ku*
    #     = K** - K*u (Kuu⁻¹ - (Lk Lkᵀ)⁻¹ Lk cov(q) Lkᵀ (Lk Lkᵀ)⁻¹) Ku*
    #     = K** - K*u (Kuu⁻¹ - Lk⁻ᵀ Lk⁻¹ Lk cov(q) Lkᵀ Lk⁻ᵀ Lk⁻¹) Ku*
    #     = K** - K*u (Kuu⁻¹ - Lk⁻ᵀ cov(q) Lk⁻¹) Ku*
    #     = K** - (K*u Lk⁻ᵀ) (Lk⁻¹ Ku*) - (K*u Lk⁻ᵀ) Lq Lqᵀ (Lk⁻¹ Ku*)
    #     = K** - A'A - (K*u Lk⁻ᵀ) Lq Lqᵀ (Lk⁻¹ Ku*)
    #     = K** - A'A - A' B B' A
    # A = Lk⁻¹ Ku*
    # NonCentered: B = Lq
    # [Centered: B = Lk⁻¹ Lq]
    q, fz = sva.q, sva.fz
    m = mean(q)
    Kuu = _chol_cov(fz)
    α = chol_lower(Kuu)' \ m
    Sv = _chol_cov(q)
    B = chol_lower(Sv)
    data = (Kuu=Kuu, B=B, α=α)
    return ApproxPosteriorGP(sva, fz.f, data)
end

function AbstractGPs.posterior(
    sva::AbstractSparseVariationalApproximation, fx::FiniteGP, ::AbstractVector{<:Real}
)
    @assert sva.fz.f === fx.f
    return posterior(sva)
end

function AbstractGPs.posterior(
    sva::AbstractSparseVariationalApproximation, lfx::LatentFiniteGP, ::Any
)
    @assert sva.fz.f === lfx.fx.f
    return posterior(sva)
end

#
# Various methods implementing the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

function Statistics.mean(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation}, x::AbstractVector
)
    return mean(f.prior, x) + cov(f.prior, x, inducing_points(f)) * f.data.α
end

# A = Lk⁻¹ Ku* is the projection matrix used in computing the predictive variance of the SparseVariationalApproximation posterior.
function _A_and_Kuf(f, x)
    Kuf = cov(f.prior, inducing_points(f), x)
    A = chol_lower(f.data.Kuu) \ Kuf
    return A, Kuf
end

_A(f, x) = first(_A_and_Kuf(f, x))

function Statistics.cov(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    return cov(f.prior, x) - At_A(A) + At_A(f.data.B' * A)
end

function Statistics.var(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation}, x::AbstractVector
)
    A = _A(f, x)
    return var(f.prior, x) - diag_At_A(A) + diag_At_A(f.data.B' * A)
end

function StatsBase.mean_and_cov(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation}, x::AbstractVector
)
    A, Kuf = _A_and_Kuf(f, x)
    μ = mean(f.prior, x) + Kuf' * f.data.α
    Σ = cov(f.prior, x) - At_A(A) + At_A(f.data.B' * A)
    return μ, Σ
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation}, x::AbstractVector
)
    A, Kuf = _A_and_Kuf(f, x)
    μ = mean(f.prior, x) + Kuf' * f.data.α
    Σ_diag = var(f.prior, x) - diag_At_A(A) + diag_At_A(f.data.B' * A)
    return μ, Σ_diag
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:AbstractSparseVariationalApproximation},
    x::AbstractVector,
    y::AbstractVector,
)
    B = f.data.B
    Ax = _A(f, x)
    Ay = _A(f, y)
    return cov(f.prior, x, y) - Ax'Ay + Ax' * B * B' * Ay
end

#
# Misc utility.
#

inducing_points(f::ApproxPosteriorGP{<:SparseVariationalApproximation}) = f.approx.fz.x

#
# elbo
#

function API.approx_lml(
    sva::AbstractSparseVariationalApproximation,
    l_fx::Union{FiniteGP,LatentFiniteGP},
    ys;
    kwargs...,
)
    return AbstractGPs.elbo(sva, l_fx, ys; kwargs...)
end

_get_prior(approx::SparseVariationalApproximation) = approx.fz.f

"""
    elbo(
        sva::AbstractSparseVariationalApproximation,
        fx::FiniteGP,
        y::AbstractVector{<:Real};
        num_data=length(y),
        quadrature=DefaultQuadrature(),
    )

Compute the Evidence Lower BOund from [1] for the process `f = fx.f ==
svgp.fz.f` where `y` are observations of `fx`, pseudo-inputs are given by `z =
svgp.fz.x` and `q(u)` is a variational distribution over inducing points `u =
f(z)`.

`quadrature` selects which method is used to calculate the expected loglikelihood in
the ELBO. The options are: `DefaultQuadrature()`, `Analytic()`, `GaussHermite()` and
`MonteCarlo()`. For likelihoods with an analytic solution, `DefaultQuadrature()` uses this
exact solution. If there is no such solution, `DefaultQuadrature()` either uses
`GaussHermite()` or `MonteCarlo()`, depending on the likelihood.

N.B. the likelihood is assumed to be Gaussian with observation noise `fx.Σy`.
Further, `fx.Σy` must be isotropic - i.e. `fx.Σy = α * I`.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.elbo(
    sva::AbstractSparseVariationalApproximation,
    fx::FiniteGP{<:AbstractGP,<:AbstractVector,<:Diagonal{<:Real,<:Fill}},
    y::AbstractVector{<:Real};
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert _get_prior(sva) === fx.f
    return _elbo(quadrature, sva, fx, y, GaussianLikelihood(fx.Σy[1]), num_data)
end

function AbstractGPs.elbo(
    ::AbstractSparseVariationalApproximation, ::FiniteGP, ::AbstractVector; kwargs...
)
    return error(
        "The observation noise fx.Σy must be homoscedastic.\n",
        "To avoid this error, construct fx using: f = GP(kernel); fx = f(x, σ²)",
        ", where σ² is a positive Real.",
    )
end

"""
    elbo(
        sva::AbstractSparseVariationalApproximation,
        lfx::LatentFiniteGP,
        y::AbstractVector;
        num_data=length(y),
        quadrature=DefaultQuadrature(),
    )

Compute the ELBO for a LatentGP with a possibly non-conjugate likelihood.
"""
function AbstractGPs.elbo(
    sva::AbstractSparseVariationalApproximation,
    lfx::LatentFiniteGP,
    y::AbstractVector;
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert _get_prior(sva) === lfx.fx.f
    return _elbo(quadrature, sva, lfx.fx, y, lfx.lik, num_data)
end

# Compute the common elements of the ELBO
function _elbo(
    quadrature::QuadratureMethod,
    sva::AbstractSparseVariationalApproximation,
    fx::FiniteGP,
    y::AbstractVector,
    lik,
    num_data::Integer,
)
    @assert _get_prior(sva) === fx.f

    f_post = posterior(sva)
    q_f = marginals(f_post(fx.x))
    variational_exp = expected_loglik(quadrature, y, q_f, lik)

    n_batch = length(y)
    scale = num_data / n_batch
    return sum(variational_exp) * scale - _prior_kl(sva)
end

_prior_kl(sva::SparseVariationalApproximation{Centered}) = kldivergence(sva.q, sva.fz)

function _prior_kl(sva::SparseVariationalApproximation{NonCentered})
    m_ε = mean(sva.q)
    C_ε = _cov(sva.q)

    # trace_term = tr(C_ε)  # does not work due to PDMat / Zygote issues
    L = chol_lower(_chol_cov(sva.q))
    trace_term = sum(L .^ 2)  # TODO remove AD workaround

    return (trace_term + m_ε'm_ε - length(m_ε) - logdet(C_ε)) / 2
end

# Pseudo-Observation Parametrisations of q(u).

@doc raw"""
    PseudoObsSparseVariationalApproximation(
        likelihood, f::AbstractGP, z::AbstractVector
    )

Parametrises `q(f(z))`, the approximate posterior at `f(z)`, using a surrogate likelihood,
`likelihood`: `q(f(z)) ∝ p(f(z)) likelihood(f(z))`.
"""
struct PseudoObsSparseVariationalApproximation{
    Tlikelihood,Tf<:AbstractGP,Tz<:AbstractVector
} <: AbstractSparseVariationalApproximation
    likelihood::Tlikelihood
    f::Tf
    z::Tz
end

_get_prior(approx::PseudoObsSparseVariationalApproximation) = approx.f

@doc raw"""
    ObsCovLikelihood(S::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})

Chooses `likelihood(u) = N(y; u, S)`. `length(y)` must be equal to the number of
pseudo-points utilised in the sparse variational approximation.
"""
struct ObsCovLikelihood{TS<:AbstractMatrix{<:Real},Ty<:AbstractVector{<:Real}}
    S::TS
    y::Ty
end

@doc raw"""
    PseudoObsSparseVariationalApproximation(
        f::AbstractGP,
        z::AbstractVector,
        S::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real},
    )

Convenience constuctor.
Equivalent to
```julia
PseudoObsSparseVariationalApproximation(ObsCovLikelihood(S, y), f, z)
```
"""
function PseudoObsSparseVariationalApproximation(
    f::AbstractGP, z::AbstractVector, S::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}
)
    return PseudoObsSparseVariationalApproximation(ObsCovLikelihood(S, y), f, z)
end

function AbstractGPs.posterior(
    approx::PseudoObsSparseVariationalApproximation{<:ObsCovLikelihood}
)
    f = approx.f
    z = approx.z
    y = approx.likelihood.y
    S = approx.likelihood.S
    return posterior(f(z, S), y)
end

function _prior_kl(approx::PseudoObsSparseVariationalApproximation{<:ObsCovLikelihood})
    f = approx.f
    z = approx.z
    y = approx.likelihood.y
    S = approx.likelihood.S

    # log marginal probability of pseudo-observations.
    logp_pseudo_obs = logpdf(f(z, S), y)

    # pseudo-reconstruction term.
    m, C = mean_and_cov(posterior(approx)(z))
    S_chol = cholesky(AbstractGPs._symmetric(S))
    quad_form = sum(abs2, S_chol.U' \ (y - m))
    pseudo_lik = -(length(y) * AbstractGPs.log2π + logdet(S_chol) + quad_form) / 2
    trace_term = tr(S_chol \ C) / 2
    return -logp_pseudo_obs + pseudo_lik - trace_term
end

@doc raw"""
    DecoupledObsCovLikelihood(
        S::AbstractMatrix{<:Real}, v::AbstractVector, y::AbstractVector{<:Real}
    )

Chooses `likelihood(u) = N(y; f(v), S)` where `length(y)` need not be equal to the number
of pseudo-points, where `f` is the GP to which this likelihood specifies the approximate
posterior over `f(z)`.
"""
struct DecoupledObsCovLikelihood{
    TS<:Diagonal{<:Real},Tv<:AbstractVector,Ty<:AbstractVector{<:Real}
}
    S::TS
    v::Tv
    y::Ty
end

@doc raw"""
    PseudoObsSparseVariationalApproximation(
        f::AbstractGP,
        z::AbstractVector,
        S::Diagonal{<:Real},
        v::AbstractVector,
        y::AbstractVector{<:Real},
    )

Convenience constructor.
Equivalent to
```julia
PseudoObsSparseVariationalApproximation(DecoupledObsCovLikelihood(S, v, y), f, z)
```
"""
function PseudoObsSparseVariationalApproximation(
    f::AbstractGP,
    z::AbstractVector,
    S::Diagonal{<:Real},
    v::AbstractVector,
    y::AbstractVector{<:Real},
)
    return PseudoObsSparseVariationalApproximation(DecoupledObsCovLikelihood(S, v, y), f, z)
end

function AbstractGPs.posterior(
    approx::PseudoObsSparseVariationalApproximation{<:DecoupledObsCovLikelihood}
)
    f = approx.f
    z = approx.z
    y = approx.likelihood.y
    S = approx.likelihood.S
    v = approx.likelihood.v
    return posterior(AbstractGPs.VFE(f(z, 1e-9)), f(v, S), y)
end

function _prior_kl(
    approx::PseudoObsSparseVariationalApproximation{<:DecoupledObsCovLikelihood}
)
    f = approx.f
    z = approx.z
    y = approx.likelihood.y
    S = approx.likelihood.S
    v = approx.likelihood.v

    # log marginal probability of pseudo-observations. Utilises DTC code.
    logp_pseudo_obs = AbstractGPs.dtc(AbstractGPs.VFE(f(z)), f(v, S), y)

    # pseudo-reconstruction term.
    m̂, Ĉ = mean_and_cov(posterior(approx)(z, 1e-18))
    At = cholesky(AbstractGPs._symmetric(cov(f(z, 1e-18)))) \ cov(f, z, v)
    m = mean(f, v) + At' * (m̂ - mean(f, z))
    pseudo_loglik = sum(map((m, s, y) -> logpdf(Normal(m, sqrt(s)), y), m, diag(S), y))
    pseudo_trace_term = sum(Ĉ .* (At * (S \ At'))) / 2
    pseudo_reconstruction = (pseudo_loglik - pseudo_trace_term)

    return -logp_pseudo_obs + pseudo_reconstruction
end

end
