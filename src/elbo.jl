"""
    elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fz::FiniteGP, q::MvNormal; n_data=1)

Compute the Evidence Lower BOund from [1] for the process `fx.f` where `y` are
observations of `fx`, pseudo-inputs are given by `z = fz.z` and `q(u)` is a
variational distribution over inducing points `u = f(z)`.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function elbo(
    fx::FiniteGP,
    y::AbstractVector{<:Real},
    fz::FiniteGP,
    q::MvNormal;
    n_data=length(y)
)
    n_batch = length(y)
    kl_term, f_mean, f_var = _elbo_intermediates(fx, fz, q)

    Σy = diag(fx.Σy) # n.b. this assumes uncorrelated observation noise
    variational_exp = expected_loglik(y, f_mean, f_var, Σy)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

function elbo(
    lfx::LatentFiniteGP,
    y::AbstractVector,
    fz::FiniteGP,
    q::MvNormal;
    n_data=length(y)
)
    n_batch = length(y)
    kl_term, f_mean, f_var = _elbo_intermediates(lfx.fx, fz, q)
    
    variational_exp = expected_loglik(y, f_mean, f_var, lfx.lik)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

# Computes the common intermediates needed for the ELBO
function _elbo_intermediates(
    fx::FiniteGP,
    fz::FiniteGP,
    q::MvNormal
)
    kl_term = kl_divergence(q, fz)
    post = approx_posterior(SVGP(), fz, q)
    f_mean, f_var = mean_and_var(post, fx.x)
    return kl_term, f_mean, f_var
end

# The closed form expected loglikelihood for a Gaussian likelihood
function expected_loglik(
    y::AbstractVector{<:Real},
    f_mean::AbstractVector,
    f_var::AbstractVector,
    Σy::AbstractVector
)
    return -0.5 * (log(2π) .+ log.(Σy) .+ ((y .- f_mean).^2 .+ f_var) ./ Σy)
end

function expected_loglik(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::BernoulliLikelihood;
    n_points=20
)
    return gauss_hermite_quadrature(y, f_mean, f_var, lik; n_points=n_points)
end

function kl_divergence(q::MvNormal, p::AbstractMvNormal)
    p_μ, p_Σ = mean(p), cov(p)
    (1/2) .* (logdet(p_Σ) - logdet(q.Σ) - length(p_μ) + tr(p_Σ \ cov(q)) +
              Xt_invA_X(cholesky(p_Σ), (q.μ - p_μ)))
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
    return (1/√π) * lls * ws
end

ChainRulesCore.@non_differentiable gausshermite(n)
