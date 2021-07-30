@testset "svgp" begin
    rng = MersenneTwister(123456)
    N_cond = 5
    N_a = 6
    N_b = 7

    # Specify prior.
    f = GP(Matern32Kernel())
    # Sample from prior.
    x = collect(range(-1.0, 1.0; length = N_cond))
    fx = f(x, 1e-15)
    y = rand(rng, fx)

    q = exact_variational_posterior(fx, fx, y)
    f_approx_post = approx_posterior(SVGP(), fx, q)

    a = collect(range(-1.0, 1.0; length = N_a))
    b = randn(rng, N_b)
    AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f_approx_post, a, b)
end
