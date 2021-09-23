function generate_data()
    Random.seed!(1)
    X = range(0, 23.5; length=48)
    fs = @. 3 * sin(10 + 0.6X) + sin(0.1X) - 1
    # invlink = normcdf
    invlink = logistic
    ps = invlink.(fs)
    Y = [rand(Bernoulli(p)) for p in ps]
    return X, Y
end

dist_y_given_f(f) = Bernoulli(logistic(f))

function build_latent_gp(theta)
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return LatentGP(GP(kernel), dist_y_given_f, 1e-8)
end

function optimize_elbo(
    build_latent_gp,
    theta0,
    xs,
    ys,
    optimizer,
    optim_options;
    newton_warmstart=true,
    newton_callback=nothing,
)
    f = similar(xs, length(xs))  # will be mutated in-place to "warm-start" the Newton steps
    objective = build_laplace_objective!(
        f, build_latent_gp, xs, ys; newton_warmstart, newton_callback
    )
    objective_grad(θ) = only(Zygote.gradient(objective, θ))

    training_results = Optim.optimize(
        objective,
        objective_grad,
        theta0,
        optimizer,
        optim_options;
        inplace=false,
    )

    lf = build_latent_gp(training_results.minimizer)
    f_post = posterior(LaplaceApproximation(; f_init=f), lf(xs), ys)
    return f_post, training_results
end

@testset "Gaussian" begin
    # should check for convergence in one step, and agreement with exact GPR
end

@testset "gradients" begin
    X, Y = generate_data()
    @testset "laplace_lml" begin
        theta0 = rand(2)
        function objective(theta)
            lf = build_latent_gp(theta)
            lml = laplace_lml(lf(X), Y)
            return -lml
        end
        fd_grad = only(FiniteDifferences.grad(central_fdm(5, 1), objective, theta0))
        ad_grad = only(Zygote.gradient(objective, theta0))
        @test ad_grad ≈ fd_grad
    end
end

@testset "chainrule" begin
    xs = [0.2, 0.3, 0.7]
    ys = [1, 1, 0]
    L = randn(3, 3)

    function newton_inner_loop_from_L(dist_y_given_f, ys, L; kwargs...)
        K = L'L
        return ApproximateGPs.newton_inner_loop(dist_y_given_f, ys, K; kwargs...)
    end

    function ChainRulesCore.frule(
        (Δself, Δdist_y_given_f, Δys, ΔL),
        ::typeof(newton_inner_loop_from_L),
        dist_y_given_f,
        ys,
        L;
        kwargs...,
    )
        K = L'L
        # K̇ = L̇'L + L'L̇
        ΔK = ΔL'L + L'ΔL
        return frule(
            (Δself, Δdist_y_given_f, Δys, ΔK),
            ApproximateGPs.newton_inner_loop,
            dist_y_given_f,
            ys,
            K;
            kwargs...,
        )
    end

    function ChainRulesCore.rrule(
        ::typeof(newton_inner_loop_from_L), dist_y_given_f, ys, L; kwargs...
    )
        K = L'L
        f_opt, newton_from_K_pullback = rrule(
            ApproximateGPs.newton_inner_loop, dist_y_given_f, ys, K; kwargs...
        )

        function newton_from_L_pullback(Δf_opt)
            (∂self, ∂dist_y_given_f, ∂ys, ∂K) = newton_from_K_pullback(Δf_opt)
            # Re⟨K̄, K̇⟩ = Re⟨K̄, L̇'L + L'L̇⟩
            # = Re⟨K̄, L̇'L⟩ + Re⟨K̄, L'L̇⟩
            # = Re⟨K̄L', L̇'⟩ + Re⟨LK̄, L̇⟩
            # = Re⟨LK̄', L̇⟩ + Re⟨LK̄, L̇⟩
            # = Re⟨LK̄' + LK̄, L̇⟩
            # = Re⟨L̄, L̇⟩
            # L̄ = L(K̄' + K̄)
            ∂L = @thunk(L * (∂K' + ∂K))

            return (∂self, ∂dist_y_given_f, ∂ys, ∂L)
        end

        return f_opt, newton_from_L_pullback
    end

    fkwargs = (; f_init=zero(ys), maxiter=100)
    test_frule(newton_inner_loop_from_L, dist_y_given_f, ys, L; fkwargs)
    test_rrule(newton_inner_loop_from_L, dist_y_given_f, ys, L; fkwargs)
end

@testset "optimization" begin
    X, Y = generate_data()
    theta0 = [0.0, 1.0]

    @testset "reference optimum" begin
        function objective(theta)
            lf = build_latent_gp(theta)
            return -laplace_lml(lf(X), Y)
        end

        @testset "NelderMead" begin
            expected_thetahat = [7.708967951453345, 1.5182348363613536]

            res = Optim.optimize(objective, theta0, NelderMead())
            #@info res

            @test res.minimizer ≈ expected_thetahat
        end

        @testset "gradient-based" begin
            expected_thetahat = [7.709076337653239, 1.51820292019697]

            objective_grad(θ) = only(Zygote.gradient(objective, θ))
            res = Optim.optimize(objective, objective_grad, theta0, LBFGS(); inplace=false)
            #@info res

            @test res.minimizer ≈ expected_thetahat
        end
    end

    @testset "warmstart vs coldstart" begin
        args = (build_latent_gp, theta0, X, Y, LBFGS(), Optim.Options(; iterations=1000))

        n_newton_coldstart = 0
        count_coldstart!(_, _) = (n_newton_coldstart += 1)

        _, res_cold = optimize_elbo(
            args...; newton_warmstart=false, newton_callback=count_coldstart!
        )
        #@info "Coldstart:\n$res_cold"

        n_newton_warmstart = 0
        count_warmstart!(_, _) = (n_newton_warmstart += 1)

        _, res_warm = optimize_elbo(
            args...; newton_warmstart=true, newton_callback=count_warmstart!
        )
        #@info "Warmstart:\n$res_warm"

        @info "Newton steps: $n_newton_coldstart (coldstart) vs $n_newton_warmstart (warmstart)"
        @test n_newton_coldstart - n_newton_warmstart > 100
        @test res_cold.minimizer ≈ res_warm.minimizer
    end
end
