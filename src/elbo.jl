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

N.B. the likelihood is assumed to be Gaussian with observation noise `fx.Σy`.
Further, `fx.Σy` must be homoscedastic and uncorrelated - i.e. `fx.Σy = α * I`.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.elbo(
    fx::FiniteGP{<:AbstractGP, <:AbstractVector, <:Diagonal{<:Real, <:Fill}},
    y::AbstractVector{<:Real},
    fz::FiniteGP,
    q::AbstractMvNormal;
    n_data=length(y),
    method=Default()
)
    return _elbo(method, fx, y, fz, q, GaussianLikelihood(fx.Σy[1]), n_data)
end

function AbstractGPs.elbo(::FiniteGP, ::AbstractVector, ::FiniteGP, ::AbstractMvNormal; kwargs...)
    return error("The observation noise fx.Σy must be homoscedastic.\n To avoid this error, construct fx using: f = GP(kernel); fx = f(x, σ²)")
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
    q_f = marginals(post(fx.x))
    variational_exp = expected_loglik(method, y, q_f, lik)

    kl_term = kldivergence(q, fz)

    n_batch = length(y)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

"""
    expected_loglik(method::ExpectationMethod, y::AbstractVector, q_f::AbstractVector{<:Normal}, lik)

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
[`elbo`](@ref)). The marginal distributions of `q(f)` are given by `q_f`.

`method` determines which method is used to calculate the expected log
likelihood - see [`elbo`](@ref) for more details.

# Extended help

`q(f)` is assumed to be an `MvNormal` distribution and `p(y | f)` is assumed to
have independent marginals such that only the marginals of `q(f)` are required.
"""
expected_loglik(method, y, q_f, lik)

"""
    expected_loglik(method::ExpectationMethod, y::AbstractVector, q_f::AbstractVector{<:Normal}, lik::ScalarLikelihood)

The expected log likelihood for a `ScalarLikelihood`, computed via `method`.
Defaults to a closed form solution if it exists, otherwise defaults to
Gauss-Hermite quadrature.
"""
function expected_loglik(
    ::Default,
    y::AbstractVector,
    q_f::AbstractVector{<:Normal},
    lik::ScalarLikelihood
)
    method = _default_method(lik)
    expected_loglik(method, y, q_f, lik)
end

# The closed form solution for independent Gaussian noise
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    q_f::AbstractVector{<:Normal},
    lik::GaussianLikelihood
)
    return sum(-0.5 * (log(2π) .+ log.(lik.σ²) .+ ((y .- mean.(q_f)).^2 .+ var.(q_f)) / lik.σ²))
end

# The closed form solution for a Poisson likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector,
    q_f::AbstractVector{<:Normal},
    ::PoissonLikelihood{ExpLink}
    )
    f_μ = mean.(q_f)
    return sum((y .* f_μ) - exp.(f_μ .+ (var.(q_f) / 2)) - loggamma.(y .+ 1))
end

# The closed form solution for an Exponential likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    q_f::AbstractVector{<:Normal},
    ::ExponentialLikelihood{ExpLink}
    )
    f_μ = mean.(q_f)
    return sum(-f_μ - y .* exp.((var.(q_f) / 2) .- f_μ))
end

# The closed form solution for a Gamma likelihood with an exponential inverse link function
function expected_loglik(
    ::Analytic,
    y::AbstractVector{<:Real},
    q_f::AbstractVector{<:Normal},
    lik::GammaLikelihood{<:Any, ExpLink}
    )
    f_μ = mean.(q_f)
    return sum((lik.α - 1) * log.(y) .- y .* exp.((var.(q_f) / 2) .- f_μ)
               .- lik.α * f_μ .- loggamma(lik.α))
end

function expected_loglik(
    ::Analytic,
    y::AbstractVector,
    q_f::AbstractVector{<:Normal},
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
    q_f::AbstractVector{<:Normal},
    lik::ScalarLikelihood
)
    # take 'n_samples' reparameterised samples
    f_μ = mean.(q_f)
    fs = f_μ .+ std.(q_f) .* randn(eltype(f_μ), length(q_f), mc.n_samples)
    lls = loglikelihood.(lik.(fs), y)
    return sum(lls) / mc.n_samples
end

function expected_loglik(
    gh::Quadrature,
    y::AbstractVector,
    q_f::AbstractVector{<:Normal},
    lik::ScalarLikelihood
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see eg. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(gh.n_points)
    # size(fs): (length(y), n_points)
    fs = √2 * std.(q_f) .* transpose(xs) .+ mean.(q_f)
    lls = loglikelihood.(lik.(fs), y)
    return sum((1/√π) * lls * ws)
end

ChainRulesCore.@non_differentiable gausshermite(n)

AnalyticLikelihood = Union{
    PoissonLikelihood,
    GaussianLikelihood,
    ExponentialLikelihood,
    GammaLikelihood
}
_default_method(::AnalyticLikelihood) = Analytic()
_default_method(_) = Quadrature()
