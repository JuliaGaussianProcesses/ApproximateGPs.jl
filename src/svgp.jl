struct SVGP end # TODO: should probably just be VFE?

function approx_posterior(::SVGP, fu::FiniteGP, q::MvNormal)
    m, A = q.μ, cholesky(q.Σ)
    Kuu = cholesky(Symmetric(cov(fu)))
    B = Kuu.L \ A.L  
    data = (A=A, m=m, Kuu=Kuu, B=B, α=Kuu \ m, u=fu.x)
    return ApproxPosteriorGP(SVGP(), fu.f, data)
end

function Statistics.var(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    # TODO: Don't compute the full covar
    return diag(cov(f, x))
end

function Statistics.mean(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    return cov(f.prior, x, f.data.u) * f.data.α
end

function Statistics.cov(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    Kuu = f.data.Kuu
    B = f.data.B
    D = f.data.Kuu.L \ Cux
    return cov(f.prior, x) - D'D + D' * B * B' * D 
end

function StatsBase.mean_and_cov(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    Cux = cov(f.prior, f.data.u, x)
    Kuu = f.data.Kuu
    B = f.data.B
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ = cov(f.prior, x) - D'D + D' * B * B' * D 
    return μ, Σ
end

function kl_divergence(q::MvNormal, p::AbstractMvNormal)
    (1/2) .* (logdet(cov(p)) - logdet(cov(q)) - length(mean(p)) + tr(cov(p) \ cov(q)) +
              AbstractGPs.Xt_invA_X(cholesky(q.Σ), (mean(q) - mean(p))))
end

function expected_loglik(y::AbstractVector{<:Real}, f_mean::AbstractVector, f_var::AbstractVector, Σy::AbstractVector)
    return -0.5 * (log(2π) .+ log.(Σy) .+ ((y .- f_mean).^2 .+ f_var) ./ Σy)
end

function elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fu::FiniteGP, q::MvNormal; n_data=1, n_batch=1)
    kl_term = kl_divergence(q, fu)
    post = approx_posterior(SVGP(), fu, q)
    f_mean = mean(post, fx.x)
    f_var = var(post, fx.x)
    Σy = diag(fx.Σy)

    variational_exp = expected_loglik(y, f_mean, f_var, Σy)
    scale = n_data / n_batch
    return sum(variational_exp) * scale - kl_term
end

