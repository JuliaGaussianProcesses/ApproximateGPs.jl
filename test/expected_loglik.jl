@testset "expected_loglik" begin
    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))

    likelihoods_to_test = [
        ExponentialLikelihood(),
        GammaLikelihood(),
        PoissonLikelihood(),
        GaussianLikelihood(),
    ]

    @testset "testing all analytic implementations" begin
        # Test that we're not missing any analytic implementation in `likelihoods_to_test`!
        implementation_types = [
            (; quadrature=m.sig.types[2], lik=m.sig.types[5]) for
            m in methods(ApproximateGPs.expected_loglik)
        ]
        analytic_likelihoods = [
            m.lik for m in implementation_types if
            m.quadrature == ApproximateGPs.Analytic && m.lik != Any
        ]
        for lik_type in analytic_likelihoods
            @test any(lik isa lik_type for lik in likelihoods_to_test)
        end
    end

    @testset "$(nameof(typeof(lik)))" for lik in likelihoods_to_test
        methods = [GaussHermite(100), MonteCarlo(1e7)]
        def = ApproximateGPs._default_quadrature(lik)
        if def isa Analytic
            push!(methods, def)
        end
        y = rand.(rng, lik.(zeros(10)))

        results = map(m -> ApproximateGPs.expected_loglik(m, y, q_f, lik), methods)
        @test all(x -> isapprox(x, results[end]; atol=1e-6, rtol=1e-3), results)
    end

    @test ApproximateGPs.expected_loglik(MonteCarlo(), zeros(5), q_f, GaussianLikelihood()) isa Real
    @test ApproximateGPs.expected_loglik(GaussHermite(), zeros(5), q_f, GaussianLikelihood()) isa Real
    @test ApproximateGPs._default_quadrature(θ -> Normal(0, θ)) isa GaussHermite
end
