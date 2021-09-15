"""
    elbo(svgp::SVGP, fx::FiniteGP, y::AbstractVector{<:Real}; num_data=length(y), quadrature=DefaultQuadrature())

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
    svgp::SVGP,
    fx::FiniteGP{<:AbstractGP,<:AbstractVector,<:Diagonal{<:Real,<:Fill}},
    y::AbstractVector{<:Real};
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert svgp.fz.f === fx.f
    return _elbo(quadrature, svgp, fx, y, GaussianLikelihood(fx.Σy[1]), num_data)
end

function AbstractGPs.elbo(::SVGP, ::FiniteGP, ::AbstractVector; kwargs...)
    return error(
        "The observation noise fx.Σy must be homoscedastic.\n To avoid this error, construct fx using: f = GP(kernel); fx = f(x, σ²)",
    )
end

"""
    elbo(svgp, ::SVGP, lfx::LatentFiniteGP, y::AbstractVector; num_data=length(y), quadrature=DefaultQuadrature())

Compute the ELBO for a LatentGP with a possibly non-conjugate likelihood.
"""
function AbstractGPs.elbo(
    svgp::SVGP,
    lfx::LatentFiniteGP,
    y::AbstractVector;
    num_data=length(y),
    quadrature=DefaultQuadrature(),
)
    @assert svgp.fz.f === lfx.fx.f
    return _elbo(quadrature, svgp, lfx.fx, y, lfx.lik, num_data)
end

# Compute the common elements of the ELBO
function _elbo(
    quadrature::QuadratureMethod,
    svgp::SVGP,
    fx::FiniteGP,
    y::AbstractVector,
    lik::ScalarLikelihood,
    num_data::Integer,
)
    @assert svgp.fz.f === fx.f
    post = posterior(svgp)
    q_f = marginals(post(fx.x))
    variational_exp = expected_loglik(quadrature, y, q_f, lik)

    kl_term = KL(svgp.q, svgp.fz)

    n_batch = length(y)
    scale = num_data / n_batch
    return sum(variational_exp) * scale - kl_term
end
