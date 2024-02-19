@testset "finite_basis" begin
    rng = MersenneTwister(123456)
    N = 50
    x = rand(rng, 2, N);
    y = sin.(norm.(eachcol(x)))

    @testset "Verify equivalence of weight space and function space posteriors" begin
        kern = FiniteBasis(identity)
        x2 = ColVecs(rand(2, N))

        # Predict mean and covariance using weight space view
        f = GP(kern)
        fx = f(x, 0.001)
        opt_pred = mean_and_cov(posterior(fx, y)(x2))

        # Predict mean and covariance as normal
        fx2 = GP(kern + ZeroKernel())(x, 0.001)
        pred = mean_and_cov(posterior(fx2, y)(x2))

        # The two approaches should be the same
        @test all(opt_pred .≈ pred)
    end

    @testset "Verify that the RFF approximation matches the exact posterior" begin
        rng = MersenneTwister(12345)
        rbf = SqExponentialKernel()
        flat_x = rand(rng, N)
        flat_x2 = rand(rng, N)
        ffkern = FiniteBasis(RandomFourierFeature(rng, rbf, 200))

        opt_pred = mean_and_cov(posterior(GP(ffkern)(flat_x, 0.001), y)(flat_x2))
        pred = mean_and_cov(posterior(GP(rbf)(flat_x, 0.001), y)(flat_x2))
        @test all(isapprox.(opt_pred, pred; atol=1e-2))
    end
end
