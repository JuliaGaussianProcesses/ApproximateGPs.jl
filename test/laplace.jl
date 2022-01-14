@testset "laplace" begin
    function generate_data()
        X = range(0, 23.5; length=48)
        # The random number generator changed in 1.6->1.7. The following vector was generated in Julia 1.6.
        # The generating code below is only kept for illustrative purposes.
        #! format: off
        Y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        #! format: on
        # Random.seed!(1)
        # fs = @. 3 * sin(10 + 0.6X) + sin(0.1X) - 1
        # # invlink = normcdf
        # invlink = logistic
        # ps = invlink.(fs)
        # Y = [rand(Bernoulli(p)) for p in ps]
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
        objective = build_laplace_objective(
            build_latent_gp, xs, ys; newton_warmstart, newton_callback
        )
        objective_grad(θ) = only(Zygote.gradient(objective, θ))

        training_results = Optim.optimize(
            objective, objective_grad, theta0, optimizer, optim_options; inplace=false
        )

        lf = build_latent_gp(training_results.minimizer)
        f_post = posterior(LaplaceApproximation(; f_init=objective.f), lf(xs), ys)
        return f_post, training_results
    end

    @testset "predictions" begin
        rng = MersenneTwister(123456)
        N_cond = 5
        N_a = 6
        N_b = 7

        # Specify prior.
        f = GP(Matern32Kernel())
        # Sample from prior.
        x = collect(range(-1.0, 1.0; length=N_cond))
        noise_scale = 0.1
        fx = f(x, noise_scale^2)
        y = rand(rng, fx)

        jitter = 0.0  # not needed in Gaussian case
        lf = LatentGP(f, f -> Normal(f, noise_scale), jitter)
        # in Gaussian case, Laplace converges to f_opt in one step; we need the
        # second step to compute the cache at f_opt rather than f_init!
        f_approx_post = posterior(LaplaceApproximation(; maxiter=2), lf(x), y)

        @testset "AbstractGPs API" begin
            a = collect(range(-1.2, 1.2; length=N_a))
            b = randn(rng, N_b)
            AbstractGPs.TestUtils.test_internal_abstractgps_interface(
                rng, f_approx_post, a, b
            )
        end

        @testset "equivalence to exact GPR for Gaussian likelihood" begin
            f_exact_post = posterior(f(x, noise_scale^2), y)
            xt = vcat(x, randn(rng, 3))  # test at training and new points

            m_approx, c_approx = mean_and_cov(f_approx_post(xt))
            m_exact, c_exact = mean_and_cov(f_exact_post(xt))

            @test m_approx ≈ m_exact
            @test c_approx ≈ c_exact
        end
    end

    @testset "gradients" begin
        @testset "approx_lml" begin
            X, Y = generate_data()

            Random.seed!(123)
            theta0 = rand(2)
            function objective(theta)
                lf = build_latent_gp(theta)
                lml = approx_lml(LaplaceApproximation(), lf(X), Y)
                return -lml
            end
            fd_grad = only(FiniteDifferences.grad(central_fdm(5, 1), objective, theta0))
            ad_grad = only(Zygote.gradient(objective, theta0))
            @test ad_grad ≈ fd_grad rtol = 1e-6
        end

        @testset "_newton_inner_loop derivatives not defined" begin
            Random.seed!(54321)

            xs = [0.2, 0.3, 0.7]
            ys = [1, 1, 0]
            theta0 = 1.234

            function eval_newton_inner_loop(theta)
                k = with_lengthscale(Matern52Kernel(), exp(theta))
                K = kernelmatrix(k, xs)
                f, cache = LaplaceApproximationModule._newton_inner_loop(
                    dist_y_given_f, ys, K; f_init=zero(xs), maxiter=100
                )
                return f
            end

            eval_newton_inner_loop(theta0)  # forward pass works
            @test_throws ErrorException Zygote.gradient(eval_newton_inner_loop, theta0)
        end

        @testset "newton_inner_loop chain rules" begin
            Random.seed!(54321)

            xs = [0.2, 0.3, 0.7]
            ys = [1, 1, 0]
            L = randn(3, 3)

            function newton_inner_loop_from_L(dist_y_given_f, ys, L; kwargs...)
                K = L'L
                return LaplaceApproximationModule.newton_inner_loop(dist_y_given_f, ys, K; kwargs...)
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
                    LaplaceApproximationModule.newton_inner_loop,
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
                    LaplaceApproximationModule.newton_inner_loop, dist_y_given_f, ys, K; kwargs...
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

            fkwargs = (; f_init=zeros(length(ys)), maxiter=100)
            test_frule(newton_inner_loop_from_L, dist_y_given_f, ys, L; fkwargs)
            test_rrule(newton_inner_loop_from_L, dist_y_given_f, ys, L; fkwargs)
        end
    end

    @testset "optimization" begin
        X, Y = generate_data()
        theta0 = [5.0, 1.0]

        @testset "reference optimum" begin
            function objective(theta)
                lf = build_latent_gp(theta)
                return -approx_lml(LaplaceApproximation(), lf(X), Y)
            end

            @testset "NelderMead" begin
                expected_thetahat = [7.708967951453345, 1.5182348363613536]

                res = Optim.optimize(objective, theta0, NelderMead())
                #@info res

                @test res.minimizer ≈ expected_thetahat rtol = 1e-4
            end

            @testset "gradient-based" begin
                expected_thetahat = [7.709076337653239, 1.51820292019697]

                objective_grad(θ) = only(Zygote.gradient(objective, θ))
                res = Optim.optimize(
                    objective, objective_grad, theta0, LBFGS(); inplace=false
                )
                #@info res

                @test res.minimizer ≈ expected_thetahat
            end
        end

        @testset "warmstart vs coldstart" begin
            args = (
                build_latent_gp, theta0, X, Y, LBFGS(), Optim.Options(; iterations=1000)
            )

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

    @testset "laplace_steps" begin
        X, Y = generate_data()
        Random.seed!(123)
        theta0 = rand(2)
        lf = build_latent_gp(theta0)
        lfx = lf(X)

        res_array = LaplaceApproximationModule.laplace_steps(lfx, Y)
        res = res_array[end]
        @test res.q isa MvNormal
    end
end
