@testset "svgp" begin
    rng = MersenneTwister(123456)
    N_cond = 5
    M = 4
    N_a = 6
    N_b = 7

    # Specify prior.
    f = GP(Matern32Kernel())

    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    fx = f(x, 1e-15)
    y = rand(rng, fx)

    # Specify inducing variables.
    z = range(-1.0, 1.0; length=M)
    fz = f(z, 1e-6)

    # Construct approximate posterior.
    q_centred = optimal_variational_posterior(fz, fx, y)
    approx_centred = SparseVariationalApproximation(Centred(), fz, q_centred)
    f_approx_post_centred = posterior(approx_centred)

    # Check that approximate posterior is self-consistent.
    a = collect(range(-1.0, 1.0; length=N_a))
    b = randn(rng, N_b)
    TestUtils.test_internal_abstractgps_interface(rng, f_approx_post_centred, a, b)

    @testset "noncentred" begin

        # Construct optimal approximate posterior.
        q = optimal_variational_posterior(fz, fx, y)
        Cuu = cholesky(Symmetric(cov(fz)))
        q_ε = MvNormal(Cuu.L \ (mean(q) - mean(fz)), Symmetric((Cuu.L \ cov(q)) / Cuu.U))

        # Check that q_ε has been properly constructed.
        @test mean(q) ≈ mean(fz) + Cuu.L * mean(q_ε)
        @test cov(q) ≈ Cuu.L * cov(q_ε) * Cuu.U

        # Construct equivalent approximate posteriors.
        approx_non_centred = SparseVariationalApproximation(NonCentred(), fz, q_ε)
        f_approx_post_non_centred = posterior(approx_non_centred)
        TestUtils.test_internal_abstractgps_interface(rng, f_approx_post_non_centred, a, b)

        # Unit-test kl_term.
        @test isapprox(
            ApproximateGPs.kl_term(approx_non_centred, f_approx_post_non_centred),
            ApproximateGPs.kl_term(approx_centred, f_approx_post_centred);
            rtol=1e-5,
        )

        # Verify that the non-centred approximate posterior agrees with centred.
        @test mean(f_approx_post_non_centred, a) ≈ mean(f_approx_post_centred, a)
        @test cov(f_approx_post_non_centred, a, b) ≈ cov(f_approx_post_centred, a, b)
        @test elbo(approx_non_centred, fx, y) ≈ elbo(approx_centred, fx, y)
    end
end
