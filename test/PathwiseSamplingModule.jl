@testset "pathwise_sampling" begin
    rng = MersenneTwister(1453)

    kernel = 0.1 * (SqExponentialKernel() ∘ ScaleTransform(0.2))
    Σy = 1e-6
    f = GP(kernel)

    input_dims = 2
    X = ColVecs(rand(rng, input_dims, 8))

    fx = f(X, Σy)
    y = rand(fx)

    Z = X[1:4]
    fz = f(Z)

    X_test = ColVecs(rand(rng, input_dims, 3))

    num_features = 10000
    rff_wsa = build_rff_weight_space_approx(rng, input_dims, num_features)

    num_samples = 1000

    function test_single_sample_stats(ap, num_samples)
        return test_stats(ap, [pathwise_sample(rng, ap, rff_wsa) for _ in 1:num_samples])
    end

    function test_multi_sample_stats(ap, num_samples)
        return test_stats(ap, pathwise_sample(rng, ap, rff_wsa, num_samples))
    end

    function test_stats(ap, function_samples)
        y_samples = reduce(hcat, map((f) -> f(X_test), function_samples))
        m_empirical = mean(y_samples; dims=2)
        Σ_empirical =
            (y_samples .- m_empirical) * (y_samples .- m_empirical)' ./ num_samples

        @test mean(ap(X_test)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
        @test cov(ap(X_test)) ≈ Σ_empirical atol = 1e-3 rtol = 1e-3
    end

    @testset "Centered SVA" begin
        q = _optimal_variational_posterior(Centered(), fz, fx, y)
        ap = posterior(SparseVariationalApproximation(Centered(), fz, q))

        test_single_sample_stats(ap, num_samples)
        test_multi_sample_stats(ap, num_samples)
    end
    @testset "NonCentered SVA" begin
        q = _optimal_variational_posterior(NonCentered(), fz, fx, y)
        ap = posterior(SparseVariationalApproximation(NonCentered(), fz, q))

        test_single_sample_stats(ap, num_samples)
        test_multi_sample_stats(ap, num_samples)
    end
    @testset "VFE" begin
        ap = posterior(AbstractGPs.VFE(fz), fx, y)

        test_single_sample_stats(ap, num_samples)
        test_multi_sample_stats(ap, num_samples)
    end
end
