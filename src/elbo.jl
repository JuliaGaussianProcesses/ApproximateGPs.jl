"Likelihoods which take a scalar as input and return a scalar."
ScalarLikelihood = Union{
    BernoulliLikelihood,
    PoissonLikelihood,
    GaussianLikelihood,
    ExponentialLikelihood,
    GammaLikelihood
}


abstract type ExpectationMethod end
struct Default <: ExpectationMethod end
struct Analytic <: ExpectationMethod end

struct Quadrature <: ExpectationMethod
    n_points::Int
end
Quadrature() = Quadrature(20)

struct MonteCarlo <: ExpectationMethod
    n_samples::Int
end
MonteCarlo() = MonteCarlo(20)

"""
    elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fz::FiniteGP, q::AbstractMvNormal; n_data=length(y), method=Default())

Compute the Evidence Lower BOund from [1] for the process `fx.f` where `y` are
observations of `fx`, pseudo-inputs are given by `z = fz.x` and `q(u)` is a
variational distribution over inducing points `u = f(z)`.

`method` selects which method is used to calculate the expected loglikelihood in
the ELBO. The options are: `Default()`, `Analytic()`, `Quadrature()` and
`MonteCarlo()`. For likelihoods with an analytic solution, `Default()` uses this
exact solution. If there is no such solution, `Default()` either uses
`Quadrature()` or `MonteCarlo()`, depending on the likelihood.

N.B. the observation noise `fx.Σy` is assumed to be homoscedastic and
uncorrelated - i.e. only `fx.Σy[1]` is used.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.elbo(
    fx::FiniteGP,
    y::AbstractVector{<:Real},
    fz::FiniteGP,
    q::AbstractMvNormal;
    n_data=length(y),
    method=Default()
)
    return _elbo(method, fx, y, fz, q, GaussianLikelihood(fx.Σy[1]), n_data)
end

"""
    elbo(lfx::LatentFiniteGP, y::AbstractVector, fz::FiniteGP, q::AbstractMvNormal; n_data=length(y), method=Default())

Compute the ELBO for a LatentGP with a possibly non-conjugate likelihood.
"""
function AbstractGPs.elbo(
    lfx::LatentFiniteGP,
    y::AbstractVector,
    fz::FiniteGP,
    q::AbstractMvNormal;
    n_data=length(y),
    method=Default()
)
    return _elbo(method, lfx.fx, y, fz, q, lfx.lik, n_data)
end

# Compute the common elements of the ELBO
function _elbo(
    method::ExpectationMethod,
    fx::FiniteGP,
    y::AbstractVector,
    fz::FiniteGP,
    q::AbstractMvNormal,
    lik::ScalarLikelihood,
    n_data::Integer
)
    post = approx_posterior(SVGP(), fz, q)
    f_mean, f_var = mean_and_var(post, fx.x)
    variational_exp = expected_loglik(method, y, f_mean, f_var, lik)

    kl_term = StatsBase.kldivergence(q, fz)

    n_batch = length(y)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

"""
    expected_loglik(method, y, f_mean, f_var, lik)

This function computes the expected log likelihood:

```math
    ∫ q(f) log p(y | f) df
```
where `p(y | f)` is the process likelihood.

`q(f)` is an approximation to the latent function values `f` given by:
```math
    q(f) = ∫ p(f | u) q(u) du
```
where `q(u)` is the variational distribution over inducing points (see
[`elbo`](@ref)). The marginal means and variances of `q(f)` are given by
`f_mean` and `f_var` respectively.

`method` determines which method is used to calculate the expected log
likelihood - see [`elbo`](@ref) for more details.

# Extended help

`q(f)` is assumed to be an `MvNormal` distribution and `p(y | f)` is assumed to
have independent marginals such that only the marginals of `q(f)` are required.
"""
expected_loglik(method, y, f_mean, f_var, lik)

"""
    expected_loglik(method::ExpectationMethod, y::AbstractVector, f_mean::AbstractVector, f_var::AbstractVector, lik::ScalarLikelihood)

The expected log likelihood for a `ScalarLikelihood`, computed via `method`.
Defaults to a closed form solution if it exists, otherwise defaults to
Gauss-Hermite quadrature.
"""
function expected_loglik(
    ::Default,
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::ScalarLikelihood
)
    method = _default_method(lik)
    expected_loglik(method, y, f_mean, f_var, lik)
end

# The closed form solution for independent Gaussian noise
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::GaussianLikelihood
)
    return sum(-0.5 * (log(2π) .+ log.(lik.σ²) .+ ((y .- f_mean).^2 .+ f_var) / lik.σ²))
end

# The closed form solution for a Poisson likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    ::PoissonLikelihood{ExpLink}
)
    return sum((y .* f_mean) - exp.(f_mean .+ (f_var / 2)) - loggamma.(y .+ 1))
end

# The closed form solution for an Exponential likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    f_mean::AbstractVector,
    f_var::AbstractVector,
    ::ExponentialLikelihood{ExpLink}
)
    return sum(-f_mean - y .* exp.((f_var / 2) .- f_mean))
end

# The closed form solution for a Gamma likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::GammaLikelihood{<:Any, ExpLink}
)
    return sum((lik.α - 1) * log.(y) .- y .* exp.((f_var / 2) .- f_mean)
               .- lik.α * f_mean .- loggamma(lik.α))
end

function expected_loglik(
    ::Analytic,
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik
)
    return error(
        "No analytic solution exists for ", typeof(lik),
        ". Use `Default()`, `Quadrature()` or `MonteCarlo()` instead."
    )
end

function expected_loglik(
    mc::MonteCarlo,
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::ScalarLikelihood
)
    # take 'n_samples' reparameterised samples with μ=f_mean and σ²=f_var
    fs = f_mean .+ .√f_var .* randn(eltype(f_mean), length(f_mean), mc.n_samples)
    lls = loglikelihood.(lik.(fs), y)
    return sum(lls) / mc.n_samples
end

function expected_loglik(
    gh::Quadrature,
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::ScalarLikelihood
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see eg. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(gh.n_points)
    # size(fs): (length(y), n_points)
    fs = √2 * .√f_var .* transpose(xs) .+ f_mean
    lls = loglikelihood.(lik.(fs), y)
    return sum((1/√π) * lls * ws)
end

ChainRulesCore.@non_differentiable gausshermite(n)

function StatsBase.kldivergence(q::AbstractMvNormal, p::AbstractMvNormal)
    p_μ, p_Σ = mean(p), cov(p)
    q_μ, q_Σ = mean(q), cov(q)
    (1/2) .* (logdet(p_Σ) - logdet(q_Σ) - length(p_μ) + tr(p_Σ \ q_Σ) +
              Xt_invA_X(cholesky(p_Σ), (q_μ - p_μ)))
end

AnalyticLikelihood = Union{
    PoissonLikelihood,
    GaussianLikelihood,
    ExponentialLikelihood,
    GammaLikelihood
}
_default_method(::AnalyticLikelihood) = Analytic()
_default_method(_) = Quadrature()
