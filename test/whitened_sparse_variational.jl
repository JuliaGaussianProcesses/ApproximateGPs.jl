@testset "whitened_sparse_variational" begin
    rng = MersenneTwister(123456)
    N_cond = 5
    N_a = 6
    N_b = 7

    # Specify prior.
    f = GP(Matern32Kernel())

    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    fx = f(x, 1e-1)
    y = rand(rng, fx)

    # Construct optimal approximate posterior.
    fz = fx
    q = optimal_variational_posterior(fz, fx, y)
    Cuu = cholesky(Symmetric(cov(fz)))
    q_ε = MvNormal(Cuu.U' \ (mean(q) - mean(fz)), Symmetric((Cuu.U' \ cov(q)) / Cuu.U))

    # Check that q_ε has been properly constructed.
    @test mean(q) ≈ mean(fz) + Cuu.U' * mean(q_ε)
    @test cov(q) ≈ Cuu.U' * cov(q_ε) * Cuu.U

    # Construct equivalent approximate posteriors.
    approx = WhitenedSparseVariationalApproximation(fz, q_ε)
    f_approx_post = posterior(approx)
    a = collect(range(-1.0, 1.0; length=N_a))
    b = randn(rng, N_b)
    AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f_approx_post, a, b)

    # Verify that the whitened approximate posterior agrees with the naive parametrisation.
    approx_naive = SparseVariationalApproximation(fz, q)
    f_approx_post_naive = posterior(approx_naive)
    @test mean(f_approx_post, a) ≈ mean(f_approx_post_naive, a)
    @test cov(f_approx_post, a, b) ≈ cov(f_approx_post_naive, a ,b)
    @test elbo(approx, fx, y) ≈ elbo(approx_naive, fx, y)
end
