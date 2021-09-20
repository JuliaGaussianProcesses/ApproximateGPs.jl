function _laplace_train_intermediates(dist_y_given_f, ys, K, f)
    # Ψ = log p(y|f) + log p(f)
    #   = loglik + log p(f)
    # dΨ/df = d_loglik - K⁻¹ f
    # at fhat: d_loglik = K⁻¹ f

    # d²Ψ/df² = d2_loglik - K⁻¹
    #         = -W - K⁻¹

    ll, d_ll, d2_ll = loglik_and_derivs(dist_y_given_f, ys, f)

    W = -Diagonal(d2_ll)
    Wsqrt = sqrt(W)
    B = I + Wsqrt * K * Wsqrt
    B_ch = cholesky(Symmetric(B))
    b = W * f + d_ll
    a = b - Wsqrt * (B_ch \ (Wsqrt * K * b))

    return (; W, Wsqrt, K, a, loglik=ll, d_loglik=d_ll, B_ch)
end

# dist_y_given_f(f) = Bernoulli(logistic(f))

function loglik_and_derivs(dist_y_given_f, ys::AbstractVector, f::AbstractVector{<:Real})
    function per_observation(fhat, y)
        ll(f) = logpdf(dist_y_given_f(f), y)
        d_ll(f) = ForwardDiff.derivative(ll, f)
        d2_ll(f) = ForwardDiff.derivative(d_ll, f)
        return ll(fhat), d_ll(fhat), d2_ll(fhat)
    end
    vec_of_l_d_d2 = map(per_observation, f, ys)
    ls = map(res -> res[1], vec_of_l_d_d2)
    ll = sum(ls)
    d_ll = map(res -> res[2], vec_of_l_d_d2)
    d2_ll = map(res -> res[3], vec_of_l_d_d2)
    return ll, d_ll, d2_ll
end

function _newton_step(dist_y_given_f, ys, K, f)
    cache = _laplace_train_intermediates(dist_y_given_f, ys, K, f)
    fnew = K * cache.a
    return fnew, cache
end

function _laplace_lml(f, c)
    return -c.a' * f / 2 + c.loglik - sum(log.(diag(c.B_ch.L)))
end

#function _laplace_lml_deriv()

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
    lml_approx = _laplace_lml(f, cache)

    return (; f, f_cov, q, lml_approx, cache)
end

function laplace_steps(dist_y_given_f, f_prior, ys; maxiter=100, f=mean(f_prior))
    @assert mean(f_prior) == zero(mean(f_prior))  # might work with non-zero prior mean but not checked
    @assert length(ys) == length(f_prior) == length(f)

    K = cov(f_prior)

    res_array = []
    for i in 1:maxiter
        @info "  - Newton iteration $i"
        fnew, cache = _newton_step(dist_y_given_f, ys, K, f)

        push!(res_array, LaplaceResult(f, fnew, cache))
        # TODO don't do all these computations unless we actually want them

        if isapprox(f, fnew)
            @info "  + converged"
            break  # converged
        else
            f = fnew
        end
    end

    return res_array
end

function laplace_posterior(lfX::AbstractGPs.LatentFiniteGP, Y; kwargs...)
    newt_res = laplace_steps(lfX.lik, lfX.fx, Y; kwargs...)
    f_post = LaplacePosteriorGP(lfX.fx, newt_res[end])
    return f_post
end

"""
laplace_lml(f::Vector, lfX::LatentFiniteGP, Y::Vector)

`f` 
"""
function laplace_lml!(f, lfX, Y)
    f_opt = Zygote.ignore() do
        newt_res = laplace_steps(lfX.lik, lfX.fx, Y; f)
        f_opt = newt_res[end].f
        f .= f_opt
        return f_opt
    end

    # TODO ideally I wouldn't copy&paste the following lines
    # but we have to re-compute this outside the Zygote.ignore() to compute gradients
    cache = _laplace_train_intermediates(lfX.lik, Y, cov(lfX.fx), f_opt)
    return _laplace_lml(f_opt, cache)
end

function laplace_lml_nonewton(f_opt, lfX, Y)
    cache = _laplace_train_intermediates(lfX.lik, Y, cov(lfX.fx), f_opt)
    return _laplace_lml(f_opt, cache)
    # d lml / d theta = \partial lml / \partial theta + \partial lml / \partial fhat \partial fhat / \partial theta
end

NEWTON_COUNTER = 0

function _reset()
	global NEWTON_COUNTER
	NEWTON_COUNTER = 0
end

function _newton_inner_loop(K, dist_y_given_f, ys; f_init, maxiter)
    global NEWTON_COUNTER
    f = f_init
    cache = nothing
    for i in 1:maxiter
	Zygote.ignore() do
		@info "  - Newton iteration $i: f[1:3]=$(f[1:3])"
	end
        fnew, cache = _newton_step(dist_y_given_f, ys, K, f)
	NEWTON_COUNTER += 1

        if isapprox(f, fnew)
	    Zygote.ignore() do
		    @info "  + converged"
	    end
            break  # converged
        else
            f = fnew
        end
    end
    return f, cache
end

function newton_inner_loop(K, dist_y_given_f, ys; f_init, maxiter)
    f_opt, _ = _newton_inner_loop(K, dist_y_given_f, ys; f_init, maxiter)
    return f_opt
end

function ChainRulesCore.frule((Δself, ΔK, Δdist_y_given_f, Δys), ::typeof(newton_inner_loop), K, dist_y_given_f, ys; f_init, maxiter)
    f_opt, cache = _newton_inner_loop(K, dist_y_given_f, ys; f_init, maxiter)

    # f = K grad_log_p_y_given_f(f)
    # fdot = Kdot grad_log_p_y_given_f(f) + K grad2_log_p_y_given_f(f) fdot
    # fdot (I - K grad2_log_p_y_given_f(f)) = Kdot grad_log_p_y_given_f(f)
    # fdot = (I - K grad2_log_p_y_given_f(f))⁻¹ Kdot grad_log_p_y_given_f(f)
    # (I - K grad2_log_p_y_given_f(f)) = (I + K W) = (√W)⁻¹ (I + √W K √W) √W = (√W)⁻¹ B √W
    # fdot = (√W)⁻¹ B⁻¹ √W Kdot grad_log_p_y_given_f(f)
    ∂f_opt = cache.Wsqrt \ (cache.B_ch \ (cache.Wsqrt * (ΔK * cache.d_loglik)))

    @info "Hit frule"

    return f_opt, (∂f_opt, NoTangent())
end

function ChainRulesCore.rrule(::typeof(newton_inner_loop), K, dist_y_given_f, ys; f_init, maxiter)
    @info "Hit rrule"
    f_opt, cache = _newton_inner_loop(K, dist_y_given_f, ys; f_init, maxiter)

    # f = K grad_log_p_y_given_f(f)
    # fdot = Kdot grad_log_p_y_given_f(f) + K grad2_log_p_y_given_f(f) fdot
    # fdot (I - K grad2_log_p_y_given_f(f)) = Kdot grad_log_p_y_given_f(f)
    # fdot = (I - K grad2_log_p_y_given_f(f))⁻¹ Kdot grad_log_p_y_given_f(f)
    # (I - K grad2_log_p_y_given_f(f)) = (I + K W) = (√W)⁻¹ (I + √W K √W) √W = (√W)⁻¹ B √W
    # fdot = (√W)⁻¹ B⁻¹ √W Kdot grad_log_p_y_given_f(f)

    # ∂f_opt = cache.Wsqrt \ (cache.B_ch \ (cache.Wsqrt * (ΔK * cache.d_loglik)))

    # Re<Δf, fdot> = Re<Δf, Wsqrt\inv B\inv Wsqrt Kdot d_loglik>
    #              = Re<Wsqrt' Binv' Wsqrt\inv' Δf d_loglik', Kdot >
    #
    # ΔK = Wsqrt * cache.B_ch' \ Wsqrt \ Δf_opt * cache.d_loglik'

    function newton_pullback(Δf_opt)
        ∂self = NoTangent()

        # ΔK = cache.B_ch' \ Δf_opt * cache.d_loglik'
	dfopt_dK_times_Δfopt = @thunk(cache.Wsqrt * (cache.B_ch \ (cache.Wsqrt \ Δf_opt)) * cache.d_loglik')

        dfopt_dlik_times_Δfopt = @not_implemented(
            "gradient of lml w.r.t. likelihood parameters"
        )
        dfopt_dY_times_Δfopt = @not_implemented("gradient of lml w.r.t. observations")
        return (∂self, dfopt_dK_times_Δfopt, dfopt_dlik_times_Δfopt, dfopt_dY_times_Δfopt)
    end

    return f_opt, newton_pullback
end

#function _newton_inner_loop_dual(::Type{T}, dual_args...) where {T<:Dual}
#    ȧrgs = (NO_FIELDS,  partials.(dual_args)...)
#    args = (_newton_inner_loop, value.(dual_args)...)
#    y, ẏ = frule(ȧrgs, args...)
#    T(y, ẏ)
#end
#
#function __newton_inner_loop(ϵ1::T1, ϵ2::T2, θ::T3) where {T1,T2,T3}
#    T = promote_type(T1, T2, T3)
#    if T <: Dual
#        _solve_dual(T, ϵ1, ϵ2, θ)
#    else
#        _solve(ϵ1, ϵ2, θ)
#    end
#end


#function ChainRulesCore.rrule(
#    config::RuleConfig{>:HasReverseMode},
#    ::typeof(newton_inner_loop),
#    K,
#    dist_y_given_f,
#    Y;
#    f_init,
#    maxiter,
#)
#    fopt = newton_inner_loop(K, dist_y_given_f, Y; f_init, maxiter)
#    cache = _laplace_train_intermediates(dist_y_given_f, Y, K, fopt)
#
#    function newton_pullback(Δfopt)
#        δself = NoTangent()
#        function newton_single_step(K, dist_y_given_f, ys)
#            fnew, cache = _newton_step(dist_y_given_f, ys, K, fopt)
#            return fnew
#        end
#
#        fopt2, d_fopt = rrule_via_ad(config, newton_single_step, K, dist_y_given_f, Y)
#        return d_fopt(Δfopt)
#        dfopt_dK_times_Δfopt = d_fopt(Δfopt)[2]
#
#        dfopt_dlik_times_Δfopt = @not_implemented(
#            "gradient of lml w.r.t. likelihood parameters"
#        )
#        dfopt_dY_times_Δfopt = @not_implemented("gradient of lml w.r.t. observations")
#        return (δself, dfopt_dK_times_Δfopt, dfopt_dlik_times_Δfopt, dfopt_dY_times_Δfopt)
#    end
#
#    return (fnew, cache), newton_pullback
#end

function laplace_lml(K, dist_y_given_f, Y; f_init=zeros(length(Y)), maxiter)
    f_opt = newton_inner_loop(K, dist_y_given_f, Y; f_init, maxiter)
    cache = _laplace_train_intermediates(dist_y_given_f, Y, K, f_opt)
    return _laplace_lml(f_opt, cache)
end

#function ChainRulesCore.rrule(::typeof(laplace_lml), K, dist_y_given_f, Y)
#    newt_res = laplace_steps(lfX.lik, lfX.fx, Y)
#    f_opt = newt_res[end].f
#    cache = newt_res[end].cache
#    lml = _laplace_lml(f_opt, cache)
#
#    function laplace_lml_pullback(Δlml)
#        ..._d(lfX.fx.f)
#	[..._d("lfx.lik")]
#        dlml_dlfX = ?
#        return NoTangent(), dlml_dlfX * Δlml, @not_implemented("gradient of lml @not_implemented("gradient of lml w.r.t. observations")
#    end
#
#    return lml, laplace_lml_pullback
#end

function optimize_elbo(build_latent_gp, theta0, X, Y, optimizer, optim_options, newton_warmstart=true)
    lf = build_latent_gp(theta0)
    f = mean(lf(X).fx)  # will be mutated in-place to "warm-start" the Newton steps

    function objective(theta)
        Zygote.ignore() do
            # Zygote does not like the try/catch within @info
            @info "Hyperparameters: $theta"
        end
        lf = build_latent_gp(theta)
        #lml = laplace_lml!(f, lf(X), Y)
	lfX = lf(X)
	K = cov(lfX.fx)
	dist_y_given_f = lfX.lik
        f_opt = newton_inner_loop(K, dist_y_given_f, Y; f_init=f, maxiter=100)
	cache = _laplace_train_intermediates(dist_y_given_f, Y, K, f_opt)
        lml = _laplace_lml(f_opt, cache)
	Zygote.ignore() do
		if newton_warmstart
			f .= f_opt
		end
	end
        return -lml
    end

    training_results = Optim.optimize(
        objective,
        θ -> only(Zygote.gradient(objective, θ)),
        theta0,
        optimizer,
        optim_options;
        inplace=false,
    )
    #training_results = Optim.optimize(
    #    objective, theta0, optimizer, optim_options;
    #    inplace=false,
    #)

    lf = build_latent_gp(training_results.minimizer)
    f_post = laplace_posterior(lf(X), Y; f)
    return f_post, training_results
end

struct LaplacePosteriorGP{Tprior,Tdata} <: AbstractGPs.AbstractGP
    prior::Tprior  # this is lfx.fx; should we store lfx itself (including lik) instead?
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
    f_var = var(f.prior.f, x) - vec(sum(v .^ 2; dims=1))
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
