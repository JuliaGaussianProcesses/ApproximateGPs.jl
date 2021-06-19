struct SVGP end # TODO: should probably just be VFE?

function approx_posterior(::SVGP, fu::FiniteGP, q::MvNormal)
    m, A = q.μ, q.Σ.chol
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
    return cov(f.prior, x) - D' * B * B' * D
end

function StatsBase.mean_and_cov(f::ApproxPosteriorGP{SVGP}, x::AbstractVector)
    # TODO: implement properly
    return mean(f, x), cov(f, x)
end

function kl_divergence(q::MvNormal, p::AbstractMvNormal)
    (1/2) * (logdet(q.Σ.chol)
             - logdet(cov(p)) - length(mean(p))
             + tr(inv(q.Σ.chol) * cov(p)) + Xt_invA_X(q.Σ.chol, (mean(q)-mean(p))))
end

function elbo(fx::FiniteGP, y::AbstractVector{<:Real}, fu::FiniteGP, q::MvNormal)
    kl_term = kl_divergence(q, fu)
    post = approx_posterior(SVGP(), fu, q)
    f_mean = mean(post, fx.x)
    f_var = var(post, fx.x)

    Σy = diag(fx.Σy)

    # TODO: general method for likelihoods - quadrature like GPFlow?
    variational_exp = -0.5 * (log(2π) .+ log.(Σy) .+ ((y .- f_mean).^2 .+ f_var) ./ Σy)

    # TODO: rescale for minibatches
    return sum(variational_exp) - kl_term
end

