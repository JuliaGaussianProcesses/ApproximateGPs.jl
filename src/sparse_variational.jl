"""
    SparseVariationalApproximation(fz::FiniteGP, q::AbstractMvNormal)

Packages the prior over the pseudo-points, `fz`, and the approximate posterior at the
pseudo-points, `q`, together into a single object.
"""
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

#
# Code below this point just implements the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

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


"""
    elbo(svgp::SparseVariationalApproximation, fx::FiniteGP, y::AbstractVector{<:Real}; num_data=length(y), quadrature=DefaultQuadrature())

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
    sva::SparseVariationalApproximation,
    fx::FiniteGP{<:AbstractGP,<:AbstractVector,<:Diagonal{<:Real,<:Fill}},
    y::AbstractVector{<:Real};
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert sva.fz.f === fx.f
    return _elbo(quadrature, sva, fx, y, GaussianLikelihood(fx.Σy[1]), num_data)
end

function AbstractGPs.elbo(
    ::SparseVariationalApproximation, ::FiniteGP, ::AbstractVector; kwargs...
)
    return error(
        "The observation noise fx.Σy must be homoscedastic.\n To avoid this error, construct fx using: f = GP(kernel); fx = f(x, σ²)",
    )
end

"""
    elbo(svgp, ::SparseVariationalApproximation, lfx::LatentFiniteGP, y::AbstractVector; num_data=length(y), quadrature=DefaultQuadrature())

Compute the ELBO for a LatentGP with a possibly non-conjugate likelihood.
"""
function AbstractGPs.elbo(
    sva::SparseVariationalApproximation,
    lfx::LatentFiniteGP,
    y::AbstractVector;
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert sva.fz.f === lfx.fx.f
    return _elbo(quadrature, sva, lfx.fx, y, lfx.lik, num_data)
end

# Compute the common elements of the ELBO
function _elbo(
    quadrature::QuadratureMethod,
    sva::SparseVariationalApproximation,
    fx::FiniteGP,
    y::AbstractVector,
    lik,
    num_data::Integer,
)
    @assert sva.fz.f === fx.f
    post = posterior(sva)
    q_f = marginals(post(fx.x))
    variational_exp = expected_loglik(quadrature, y, q_f, lik)

    kl_term = KL(sva.q, sva.fz)

    n_batch = length(y)
    scale = num_data / n_batch
    return sum(variational_exp) * scale - kl_term
end
