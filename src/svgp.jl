struct SVGP end

function approx_posterior(::SVGP, fu::FiniteGP, q::MvNormal)
    m, A = mean(q), cholesky(cov(q))
    Kuu = cholesky(Symmetric(cov(fu)))
    B = Kuu.L \ A.L
    data = (A=A, m=m, Kuu=Kuu, B=B, α=Kuu \ m, u=fu.x)
    return ApproxPosteriorGP(SVGP(), fu.f, data)
end

function Statistics.var(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    D = f.data.Kuu.L \ Cux
    return var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D) 
end

function Statistics.mean(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    return cov(f.prior, x, f.data.u) * f.data.α
end

function Statistics.cov(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    D = f.data.Kuu.L \ Cux
    return cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D) 
end

function StatsBase.mean_and_cov(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ = cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D) 
    return μ, Σ
end

function StatsBase.mean_and_var(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ_diag = var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D) 
    return μ, Σ_diag
end

function kl_divergence(q::MvNormal, p::AbstractMvNormal)
    p_μ, p_Σ = mean(p), cov(p)
    (1/2) .* (logdet(p_Σ) - logdet(q.Σ) - length(p_μ) + tr(p_Σ \ q.Σ) +
              Xt_invA_X(cholesky(q.Σ), (q.μ - p_μ)))
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

function elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fu::FiniteGP, q::MvNormal; n_data=1, n_batch=1)
    kl_term = kl_divergence(q, fu)
    post = approx_posterior(SVGP(), fu, q)
    f_mean, f_var = mean_and_var(post, fx.x)
    Σy = diag(fx.Σy)

    variational_exp = expected_loglik(y, f_mean, f_var, Σy)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

function elbo(fx::LatentFiniteGP, y::AbstractVector{<:Real}, fu::FiniteGP, q::MvNormal; n_data=1, n_batch=1)
    kl_term = kl_divergence(q, fu)
    post = approx_posterior(SVGP(), fu, q)
    f_mean, f_var = mean_and_var(post, fx.fx.x)
    
    variational_exp = expected_loglik(y, f_mean, f_var, fx.lik)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

