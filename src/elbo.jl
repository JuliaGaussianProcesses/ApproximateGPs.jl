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
    @assert sva.fz.f === fx.f
    return _elbo(quadrature, sva, fx, y, GaussianLikelihood(fx.Σy[1]), num_data)
end

function AbstractGPs.elbo(
    ::AbstractSparseVariationalApproximation, ::FiniteGP, ::AbstractVector; kwargs...
)
    return error(
        "The observation noise fx.Σy must be homoscedastic.\n To avoid this error, construct fx using: f = GP(kernel); fx = f(x, σ²)",
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
    @assert sva.fz.f === lfx.fx.f
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
    @assert sva.fz.f === fx.f
    post = posterior(sva)
    q_f = marginals(post(fx.x))
    variational_exp = expected_loglik(quadrature, y, q_f, lik)

    n_batch = length(y)
    scale = num_data / n_batch
    return sum(variational_exp) * scale - kl_term(sva, post)
end

kl_term(sva::SparseVariationalApproximation, post) = KL(sva.q, sva.fz)

function kl_term(sva::WhitenedSparseVariationalApproximation, post)
    m_ε = mean(sva.q_ε)
    return (tr(cov(sva.q_ε)) + m_ε'm_ε - length(m_ε) - logdet(post.data.C_ε)) / 2
end
