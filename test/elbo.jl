@testset "elbo" begin
    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    f_mean = rand(rng, 10)
    f_var = rand(rng, 10)

    @testset "$lik" for lik in Base.uniontypes(SparseGPs.ScalarLikelihood)
        l = lik()
        methods = [Quadrature(100), MonteCarlo(1000000)]
        def = SparseGPs._default_method(l)
        if def isa Analytic push!(methods, def) end
        y = rand.(rng, l.(f_mean))

        results = map(m -> SparseGPs.expected_loglik(m, y, f_mean, f_var, l), methods)
        @test all(x->isapprox(x, results[end], rtol=1e-3), results)
    end
end
