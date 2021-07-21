"Likelihoods which take a scalar (or vector of scalars) as input and return a single scalar."
ScalarLikelihood = Union{BernoulliLikelihood,PoissonLikelihood,GaussianLikelihood}

"""
    elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fz::FiniteGP, q::AbstractMvNormal; n_data=length(y))

Compute the Evidence Lower BOund from [1] for the process `fx.f` where `y` are
observations of `fx`, pseudo-inputs are given by `z = fz.x` and `q(u)` is a
variational distribution over inducing points `u = f(z)`.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function elbo(
    fx::FiniteGP,
    y::AbstractVector{<:Real},
    fz::FiniteGP,
    q::AbstractMvNormal;
    n_data=length(y),
    method=:default,
    kwargs...
)
    return _elbo(fx, y, fz, q, fx.Σy, n_data, method; kwargs...)
end


"""
    elbo(lfx::LatentFiniteGP, y::AbstractVector, fz::FiniteGP, q::AbstractMvNormal; n_data=length(y))

Compute the ELBO for a LatentGP with a possibly non-conjugate likelihood.
"""
function elbo(
    lfx::LatentFiniteGP,
    y::AbstractVector,
    fz::FiniteGP,
    q::AbstractMvNormal;
    n_data=length(y),
    method=:default,
    kwargs...
)
    return _elbo(lfx.fx, y, fz, q, lfx.lik, n_data, method; kwargs...)
end


function _elbo(
    fx::FiniteGP,
    y::AbstractVector,
    fz::FiniteGP,
    q::AbstractMvNormal,
    lik::Union{AbstractVecOrMat,ScalarLikelihood},
    n_data::Integer,
    method::Symbol;
    kwargs...
)
    post = approx_posterior(SVGP(), fz, q)
    f_mean, f_var = mean_and_var(post, fx.x)
    variational_exp = expected_loglik(y, f_mean, f_var, lik; method, kwargs...)

    kl_term = StatsBase.kldivergence(q, fz)

    n_batch = length(y)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

"""
    expected_loglik(y, f_mean, f_var, [Σy | lik])

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
[`elbo`](@ref)).

Where possible, this expectation is calculated in closed form. Otherwise, it is
approximated using Gauss-Hermite quadrature by default.

# Extended help

`q(f)` is assumed to be an `MvNormal` distribution and `p(y | f)` is assumed to
have independent marginals such that only the marginals of `q(f)` are required.
"""
function expected_loglik end

"""
    expected_loglik(y::AbstractVector{<:Real}, f_mean::AbstractVector, f_var::AbstractVector, Σy::AbstractMatrix)

The expected log likelihood for a Gaussian likelihood, computed in closed form by default.
"""
function expected_loglik(
    y::AbstractVector{<:Real},
    f_mean::AbstractVector,
    f_var::AbstractVector,
    Σy::AbstractMatrix;
    method=:default,
    kwargs...
)
    if method === :default
        return closed_form_expectation(y, f_mean, f_var, diag(Σy))
    else
        return expected_loglik(y, f_mean, f_var, GaussianLikelihood(Σy[1]); method, kwargs...)
    end
end

"""
    expected_loglik(y::AbstractVector, f_mean::AbstractVector, f_var::AbstractVector, lik::ScalarLikelihood; n_points=20)

The expected log likelihood for a `ScalarLikelihood`, approximated via
Gauss-Hermite quadrature with `n_points` quadrature points.
"""
function expected_loglik(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::ScalarLikelihood;
    method=:default,
    n_points=20,
    n_samples=20
)
    if method === :default && has_closed_form_expectation(lik)
        return closed_form_expectation(y, f_mean, f_var, lik)
    elseif method === :default || method === :gausshermite
        return gauss_hermite_quadrature(y, f_mean, f_var, lik; n_points)
    elseif method === :montecarlo
        return monte_carlo_expectation(y, f_mean, f_var, lik; n_samples)
    end
end

function closed_form_expectation(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    Σy::AbstractVector
    )
    return sum(-0.5 * (log(2π) .+ log.(Σy) .+ ((y .- f_mean).^2 .+ f_var) ./ Σy))
end

function closed_form_expectation(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    ::PoissonLikelihood
    )
    return sum(y .* f_mean - exp(f_mean .+ (f_var / 2) - loggamma.(y)))
end

function monte_carlo_expectation(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::ScalarLikelihood;
    n_samples=20
)
    # take 'n_samples' reparameterised samples with μ=f_mean and σ²=f_var
    fs = f_mean .+ .√f_var .* randn(eltype(f_mean), length(f_mean), n_samples)
    lls = loglikelihood.(lik.(fs), y)
    return sum(lls) / n_samples
end

function gauss_hermite_quadrature(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik;
    n_points=20
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see eg. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(n_points)
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

has_closed_form_expectation(lik::Union{PoissonLikelihood,GaussianLikelihood}) = true
has_closed_form_expectation(lik) = false
