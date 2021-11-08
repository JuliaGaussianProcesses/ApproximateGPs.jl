@testset "elbo" begin
    rng, N = MersenneTwister(654321), 20
    x = rand(rng, N) * 10
    y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)
    z = x[1:5]

    kernel = make_kernel([0.2, 0.6])
    f = GP(kernel)
    fx = f(x, 0.1)
    fz = f(z)
    q_ex = optimal_variational_posterior(fz, fx, y)

    sva = SparseVariationalApproximation(Centred(), fz, q_ex)
    @test elbo(sva, fx, y) isa Real
    @test elbo(sva, fx, y) ≤ logpdf(fx, y)

    fx_bad = f(x, fill(0.1, N))
    @test_throws ErrorException elbo(sva, fx_bad, y)

    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))
end
