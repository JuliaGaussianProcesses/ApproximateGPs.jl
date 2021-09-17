function _laplace_train_intermediates(overall_loglik, K, f)
    loglik = overall_loglik(f)  # TODO use withgradient instead
    d_loglik, = gradient(overall_loglik, f)
    d2_loglik, = diaghessian(overall_loglik, f)

    W = -Diagonal(d2_loglik)
    Wsqrt = sqrt(W)
    B = I + Wsqrt * K * Wsqrt
    B_ch = cholesky(Symmetric(B))
    b = W * f + d_loglik
    a = b - Wsqrt * (B_ch \ (Wsqrt * K * b))
    
    return (; W, Wsqrt, K, a, loglik, d_loglik, B_ch)
end

function _newton_step(overall_loglik, K, f)
    cache = _laplace_train_intermediates(overall_loglik, K, f)
    fnew = K * cache.a
    return fnew, cache
end

function laplace_lml(f, c)
    return -c.a' * f / 2 + c.loglik - sum(log.(diag(c.B_ch.L)))
end

function laplace_f_cov(cache)
    # (K⁻¹ + W)⁻¹
    # = (√W⁻¹) (√W⁻¹ (K⁻¹ + W) √W⁻¹)⁻¹ (√W⁻¹)
    # = (√W⁻¹) (√W⁻¹ K⁻¹ √W⁻¹ + I)⁻¹ (√W⁻¹)
    # ; (I + C⁻¹)⁻¹ = I - (I + C)⁻¹
    # = (√W⁻¹) (I - (I + √W K √W)⁻¹) (√W⁻¹)
    # = (√W⁻¹) (I - B⁻¹) (√W⁻¹)
    B_ch = cache.B_ch
    Wsqrt_inv = inv(cache.Wsqrt)
    return Wsqrt_inv * (I - inv(B_ch)) * Wsqrt_inv
end

function LaplaceResult(f, fnew, cache)
    # TODO should we use fnew?
    f_cov = laplace_f_cov(cache)
    q = MvNormal(f, AbstractGPs._symmetric(f_cov))
    lml_approx = laplace_lml(f, cache)

    return (; f, f_cov, q, lml_approx, cache)
end

function laplace_steps(dist_y_given_f, f_prior, ys; maxiter = 100, f = mean(f_prior))
    @assert mean(f_prior) == zero(mean(f_prior))  # might work with non-zero prior mean but not checked
    @assert length(ys) == length(f_prior) == length(f)

    overall_loglik(fs) = sum(logpdf(dist_y_given_f(f), y) for (f, y) in zip(fs, ys))
    K = cov(f_prior)

    res_array = []
    for i = 1:maxiter
        @info "iteration $i"
        fnew, cache = _newton_step(overall_loglik, K, f)

        push!(res_array, LaplaceResult(f, fnew, cache))
        # TODO don't do all these computations

        if isapprox(f, fnew)
            break  # converged
        else
            f = fnew
        end
    end

    return res_array
end

function laplace_posterior(lfX::AbstractGPs.LatentFiniteGP, Y)
    newt_res = laplace_steps(lfX.lik, lfX.fx, Y)
    f_post = LaplacePosteriorGP(lfX.fx, newt_res[end])
    return f_post
end

function optimize_elbo(build_latent_gp, theta0, X, Y, optimizer, optim_options)
    lf = build_latent_gp(theta0)
    lfX = lf(X)
    f = mean(lfX.fx)

    function objective(theta)
        # @info "Hyperparameters: $theta" # TODO does not work with Zygote
        lf = build_latent_gp(theta)
        lfX = lf(X)

        f_opt = Zygote.ignore() do
            newt_res = laplace_steps(lfX.lik, lfX.fx, Y; f)
            f_opt = newt_res[end].f
            f .= f_opt
            return f_opt
        end

	# TODO ideally I wouldn't copy&paste the following lines
        overall_loglik(fs) = sum(logpdf(dist_y_given_f(f), y) for (f, y) in zip(fs, Y))
        K = cov(lfX.fx)

	# but we have to re-compute this outside the Zygote.ignore() to compute gradients
        cache = _laplace_train_intermediates(overall_loglik, K, f_opt)
        return -laplace_lml(f_opt, cache)
    end

    training_results = Optim.optimize(
        objective, θ -> only(Zygote.gradient(objective, θ)), theta0, optimizer, optim_options;
        inplace=false,
    )
    
    lf = build_latent_gp(training_results.minimizer)
    lfX = lf(X)

    newt_res = laplace_steps(lfX.lik, lfX.fx, Y; f)
    f_post = LaplacePosteriorGP(lfX.fx, newt_res[end])
    return f_post
end

struct LaplacePosteriorGP{Tprior,Tdata} <: AbstractGPs.AbstractGP
    prior::Tprior
    data::Tdata
end

function _laplace_predict_intermediates(cache, prior_at_x, xnew)
    k_x_xnew = cov(prior_at_x.f, prior_at_x.x, xnew)
    f_mean = mean(prior_at_x.f, xnew) + k_x_xnew' * cache.d_loglik
    L = cache.B_ch.L
    v = L \ (cache.Wsqrt * k_x_xnew)
    return f_mean, v
end

function StatsBase.mean_and_var(f::LaplacePosteriorGP, x::AbstractVector)
    f_mean, v = _laplace_predict_intermediates(f.data.cache, f.prior, x)
    f_var = var(f.prior.f, x) - vec(sum(v .^ 2, dims = 1))
    return f_mean, f_var
end

function StatsBase.mean_and_cov(f::LaplacePosteriorGP, x::AbstractVector)
    f_mean, v = _laplace_predict_intermediates(f.data.cache, f.prior, x)
    f_cov = cov(f.prior.f, x) - v' * v
    return f_mean, f_cov
end

function Statistics.mean(f::LaplacePosteriorGP, x::AbstractVector)
    d_loglik = f.data.cache.d_loglik
    return mean(f.prior.f, x) + cov(f.prior.f, f.prior.x, x)' * d_loglik
end

function Statistics.cov(f::LaplacePosteriorGP, x::AbstractVector)
    return last(mean_and_cov(f, x))
end

function Statistics.var(f::LaplacePosteriorGP, x::AbstractVector)
    return last(mean_and_var(f, x))
end
