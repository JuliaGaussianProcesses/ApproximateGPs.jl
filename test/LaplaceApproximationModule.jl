@testset "laplace" begin
    generate_data = ApproximateGPs.TestUtils.generate_data
    dist_y_given_f = ApproximateGPs.TestUtils.dist_y_given_f
    build_latent_gp = ApproximateGPs.TestUtils.build_latent_gp

    quiet_gradient(f, x) =
        with_logger(NullLogger()) do
            only(Zygote.gradient(f, x))
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
        objective_grad(θ) = quiet_gradient(objective, θ)

        training_results = Optim.optimize(
            objective, objective_grad, theta0, optimizer, optim_options; inplace=false
        )

        lf = build_latent_gp(training_results.minimizer)
        f_post = posterior(LaplaceApproximation(; f_init=objective.cache.f), lf(xs), ys)
        return f_post, training_results
    end

    @testset "predictions" begin
        approx = LaplaceApproximation(; maxiter=2)
        ApproximateGPs.TestUtils.test_approximation_predictions(approx)
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
            ad_grad = quiet_gradient(objective, theta0)
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
            with_logger(NullLogger()) do
                @test_throws ErrorException Zygote.gradient(eval_newton_inner_loop, theta0)
            end
        end

        @testset "newton_inner_loop chain rules" begin
            Random.seed!(54321)

            xs = [0.2, 0.3, 0.7]
            ys = [1, 1, 0]
            L = randn(3, 3)

            function newton_inner_loop_from_L(dist_y_given_f, ys, L; kwargs...)
                K = L'L
                return LaplaceApproximationModule.newton_inner_loop(
                    dist_y_given_f, ys, K; kwargs...
                )
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
                    LaplaceApproximationModule.newton_inner_loop,
                    dist_y_given_f,
                    ys,
                    K;
                    kwargs...,
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

                objective_grad(θ) = quiet_gradient(objective, θ)
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

    @testset "GitHub issue #109" begin
        build_latent_gp() = LatentGP(GP(SEKernel()), BernoulliLikelihood(), 1e-8)

        x = ColVecs(randn(2, 5))
        _, y = rand(build_latent_gp()(x))

        objective = build_laplace_objective(build_latent_gp, x, y)
        _ = objective()  # check that it works
    end
end
