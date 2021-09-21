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

@testset "laplace" begin
    X, Y = generate_data()
    theta0 = [0.0, 1.0]

    function objective(theta)
        lf = build_latent_gp(theta)
        lfX = lf(X)
        f_init, K = mean_and_cov(lfX.fx)
        lml = ApproximateGPs.laplace_lml(K, lfX.lik, Y; f_init, maxiter=100)
        return -lml
    end

    @testset "NelderMead" begin
        expected_thetahat = [7.708967951453345, 1.5182348363613536]

        res = Optim.optimize(objective, theta0, NelderMead(); inplace=false)

        @test res.minimizer ≈ expected_thetahat
    end
    @testset "gradient-based" begin
        expected_thetahat = [7.709076337653239, 1.51820292019697]

        res = Optim.optimize(
            objective,
            θ -> only(Zygote.gradient(objective, θ)),
            theta0,
            LBFGS();
            inplace=false,
        )
        @info res

        @test res.minimizer ≈ expected_thetahat

        n_newton_coldstart = 0
        function count_coldstart!(_, _)
            return n_newton_coldstart += 1
        end

        _, res_cold = ApproximateGPs.optimize_elbo(
            build_latent_gp,
            theta0,
            X,
            Y,
            LBFGS(),
            Optim.Options(; iterations=1000);
            newton_warmstart=false,
            newton_callback=count_coldstart!,
        )

        n_newton_warmstart = 0
        function count_warmstart!(_, _)
            return n_newton_warmstart += 1
        end

        _, res_warm = ApproximateGPs.optimize_elbo(
            build_latent_gp,
            theta0,
            X,
            Y,
            LBFGS(),
            Optim.Options(; iterations=1000);
            newton_warmstart=true,
            newton_callback=count_warmstart!,
        )

        @test n_newton_warmstart < n_newton_coldstart
        @test res_cold.minimizer ≈ res_warm.minimizer
    end
end
