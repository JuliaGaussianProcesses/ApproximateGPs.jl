# workaround for https://github.com/JuliaDiff/ChainRulesCore.jl/issues/470 to avoid Zygote dependency
ignore_ad(closure) = closure()
@non_differentiable ignore_ad(closure)

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

    return (; K, f, W, Wsqrt, loglik=ll, d_loglik=d_ll, B_ch, a)
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

function _laplace_lml(f, cache)
    # -a' * f / 2 + loglik - sum(log.(diag(B_ch.L)))
    return -cache.a' * f / 2 + cache.loglik - sum(log.(diag(cache.B_ch.L)))
end

function _newton_inner_loop(dist_y_given_f, ys, K; f_init, maxiter, callback=nothing)
    f = f_init
    cache = nothing
    for i in 1:maxiter
        ignore_ad() do
            @debug "  - Newton iteration $i: f[1:3]=$(f[1:3])"
        end
        fnew, cache = _newton_step(dist_y_given_f, ys, K, f)
        if !isnothing(callback)
            callback(fnew, cache)
        end

        if isapprox(f, fnew)
            ignore_ad() do
                @debug "  + converged"
            end
            #f = fnew
            break  # converged
        else
            f = fnew
        end
    end
    return f, cache
end

# Currently, we have a separate function that returns only f_opt to simplify frule/rrule
function newton_inner_loop(dist_y_given_f, ys, K; f_init, maxiter, callback=nothing)
    f_opt, _ = _newton_inner_loop(dist_y_given_f, ys, K; f_init, maxiter, callback)
    return f_opt
end

function ChainRulesCore.frule(
    (Δself, Δdist_y_given_f, Δys, ΔK),
    ::typeof(newton_inner_loop),
    dist_y_given_f,
    ys,
    K;
    kwargs...,
)
    f_opt, cache = _newton_inner_loop(dist_y_given_f, ys, K; kwargs...)

    # f = K grad_log_p_y_given_f(f)
    # fdot = Kdot grad_log_p_y_given_f(f) + K grad2_log_p_y_given_f(f) fdot
    # fdot (I - K grad2_log_p_y_given_f(f)) = Kdot grad_log_p_y_given_f(f)
    # fdot = (I - K grad2_log_p_y_given_f(f))⁻¹ Kdot grad_log_p_y_given_f(f)
    # (I - K grad2_log_p_y_given_f(f)) = (I + K W) = (√W)⁻¹ (I + √W K √W) √W = (√W)⁻¹ B √W
    # fdot = (√W)⁻¹ B⁻¹ √W Kdot grad_log_p_y_given_f(f)
    ∂f_opt = cache.Wsqrt \ (cache.B_ch \ (cache.Wsqrt * (ΔK * cache.d_loglik)))

    @debug "Hit frule"

    return f_opt, ∂f_opt
end

function ChainRulesCore.rrule(::typeof(newton_inner_loop), dist_y_given_f, ys, K; kwargs...)
    @debug "Hit rrule"
    f_opt, cache = _newton_inner_loop(dist_y_given_f, ys, K; kwargs...)

    # f = K (∇log p(y|f))                               (RW 3.17)
    # δf = δK (∇log p(y|f)) + K δ(∇log p(y|f))
    #    = δK (∇log p(y|f)) + K ∇(∇log p(y|f)) δf
    # δf (I - K ∇∇log p(y|f)) = δK (∇log p(y|f))
    # δf (I + K W) = δK (∇log p(y|f))
    # δf = (I + K W)⁻¹ δK (∇log p(y|f))                 (RW 5.24)
    # (I + K W) = (√W)⁻¹ (I + √W K √W) √W = (√W)⁻¹ B √W
    # δf = (√W)⁻¹ B⁻¹ √W δK (∇log p(y|f))

    # ∂f_opt = cache.Wsqrt \ (cache.B_ch \ (cache.Wsqrt * (ΔK * cache.d_loglik)))

    # Re<Δf, δf> = Re<Δf, Wsqrt\inv B\inv Wsqrt δK d_loglik>
    #            = Re<Wsqrt' B\inv' Wsqrt\inv' Δf d_loglik', δK>
    #
    # ΔK = Wsqrt' * cache.B_ch' \ Wsqrt' \ Δf_opt * cache.d_loglik'

    function newton_pullback(Δf_opt)
        ∂self = NoTangent()

        ∂dist_y_given_f = @not_implemented(
            "gradient of Newton's method w.r.t. likelihood parameters"
        )

        ∂ys = @not_implemented("gradient of Newton's method w.r.t. observations")

        # ∂K = df/dK Δf
        ∂K = @thunk(cache.Wsqrt * (cache.B_ch \ (cache.Wsqrt \ Δf_opt)) * cache.d_loglik')

        return (∂self, ∂dist_y_given_f, ∂ys, ∂K)
    end

    return f_opt, newton_pullback
end

function laplace_lml(dist_y_given_f, ys, K, f_opt)
    cache = _laplace_train_intermediates(dist_y_given_f, ys, K, f_opt)
    return _laplace_lml(f_opt, cache)
end

function laplace_lml(
    dist_y_given_f, ys, K; f_init=zeros(length(ys)), maxiter=100, newton_kwargs...
)
    f_opt = newton_inner_loop(dist_y_given_f, ys, K; f_init, maxiter, newton_kwargs...)
    return laplace_lml(dist_y_given_f, ys, K, f_opt)
end

function laplace_lml(lfx::LatentFiniteGP, ys; newton_kwargs...)
    dist_y_given_f, K, newton_kwargs = _check_laplace_inputs(lfx, ys; newton_kwargs...)
    return laplace_lml(dist_y_given_f, ys, K; newton_kwargs...)
end

function laplace_f_and_lml(lfx::LatentFiniteGP, ys; newton_kwargs...)
    dist_y_given_f, K, newton_kwargs = _check_laplace_inputs(lfx, ys; newton_kwargs...)
    f_opt = newton_inner_loop(dist_y_given_f, ys, K; newton_kwargs...)
    lml = laplace_lml(dist_y_given_f, ys, K, f_opt)
    return f_opt, lml
end

function _check_laplace_inputs(
    lfx::LatentFiniteGP, ys; f_init=nothing, maxiter=100, newton_kwargs...
)
    fx = lfx.fx
    @assert mean(fx) == zero(mean(fx))  # might work with non-zero prior mean but not checked
    @assert length(ys) == length(fx)
    dist_y_given_f = lfx.lik
    K = cov(fx)
    if isnothing(f_init)
        f_init = mean(fx)
    end
    return dist_y_given_f, K, (; f_init, maxiter, newton_kwargs...)
end

function build_laplace_objective!(
    f,
    build_latent_gp,
    xs,
    ys;
    newton_warmstart=true,
    newton_callback=nothing,
    newton_maxiter=100,
)
    initialize_f = true

    function objective(theta)
        lf = build_latent_gp(theta)
        lfx = lf(xs)
        ignore_ad() do
            # Zygote does not like the try/catch within @info etc.
            @debug "Hyperparameters: $theta"
            if initialize_f
                f .= mean(lfx.fx)
            end
        end
        f_opt, lml = laplace_f_and_lml(
            lfx, ys; f_init=f, maxiter=newton_maxiter, callback=newton_callback
        )
        ignore_ad() do
            if newton_warmstart
                f .= f_opt
                initialize_f = false
            end
        end
        return -lml
    end

    return objective
end

function build_laplace_objective(build_latent_gp, xs, ys; kwargs...)
    f = similar(xs, length(xs))  # will be mutated in-place to "warm-start" the Newton steps
    return build_laplace_objective!(f, build_latent_gp, xs, ys; kwargs...)
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

function LaplaceResult(fnew, cache)
    f = cache.f
    f_cov = laplace_f_cov(cache)
    q = MvNormal(f, AbstractGPs._symmetric(f_cov))
    lml_approx = _laplace_lml(f, cache)

    return (; fnew, f_cov, q, lml_approx, cache...)
end

"""
    laplace_steps(lfx::LatentFiniteGP, ys; newton_kwargs...)

For demonstration purposes: returns an array of all the intermediate
approximations of each Newton step.

If you are only interested in the actual posterior, use
`posterior(::LaplaceApproximation, ...`.

TODO figure out how to get the `@ref` to work...
"""
function laplace_steps(lfx::LatentFiniteGP, ys; newton_kwargs...)
    dist_y_given_f, K, newton_kwargs = _check_laplace_inputs(lfx, ys; newton_kwargs...)

    res_array = []

    function store_result!(fnew, cache)
        return push!(res_array, LaplaceResult(fnew, cache))
    end

    _ = newton_inner_loop(dist_y_given_f, ys, K; newton_kwargs..., callback=store_result!)

    return res_array
end

struct LaplaceApproximation{Tkw}
    newton_kwargs::Tkw
end

LaplaceApproximation(; newton_kwargs...) = LaplaceApproximation(newton_kwargs)

function approx_lml(la::LaplaceApproximation, lfx::LatentFiniteGP, ys)
    return laplace_lml(lfx, ys; la.newton_kwargs...)
end

function AbstractGPs.posterior(la::LaplaceApproximation, lfx::LatentFiniteGP, ys)
    dist_y_given_f, K, newton_kwargs = _check_laplace_inputs(lfx, ys; la.newton_kwargs...)
    _, cache = _newton_inner_loop(dist_y_given_f, ys, K; newton_kwargs...)
    # TODO: should we run newton_inner_loop() and _laplace_train_intermediates() explicitly?
    f_post = ApproxPosteriorGP(la, lfx.fx, cache)
    # TODO: instead of lfx.fx, should we store lfx itself (including lik)?
    return f_post
end

const LaplacePosteriorGP = ApproxPosteriorGP{<:LaplaceApproximation}

function _laplace_predict_intermediates(cache, prior_at_x, xnew)
    k_x_xnew = cov(prior_at_x.f, prior_at_x.x, xnew)
    f_mean = mean(prior_at_x.f, xnew) + k_x_xnew' * cache.d_loglik
    L = cache.B_ch.L
    v = L \ (cache.Wsqrt * k_x_xnew)
    return f_mean, v
end

function StatsBase.mean_and_var(f::LaplacePosteriorGP, x::AbstractVector)
    f_mean, v = _laplace_predict_intermediates(f.data, f.prior, x)
    f_var = var(f.prior.f, x) - vec(sum(v .^ 2; dims=1))
    return f_mean, f_var
end

function StatsBase.mean_and_cov(f::LaplacePosteriorGP, x::AbstractVector)
    f_mean, v = _laplace_predict_intermediates(f.data, f.prior, x)
    f_cov = cov(f.prior.f, x) - v' * v
    return f_mean, f_cov
end

function Statistics.mean(f::LaplacePosteriorGP, x::AbstractVector)
    d_loglik = f.data.d_loglik
    return mean(f.prior.f, x) + cov(f.prior.f, f.prior.x, x)' * d_loglik
end

function Statistics.cov(f::LaplacePosteriorGP, x::AbstractVector)
    return last(mean_and_cov(f, x))
end

function Statistics.var(f::LaplacePosteriorGP, x::AbstractVector)
    return last(mean_and_var(f, x))
end
