@testset "expected_loglik" begin
    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))

    likelihoods_to_test = [
        ExponentialLikelihood, GammaLikelihood, PoissonLikelihood, GaussianLikelihood
    ]

    @testset "testing all analytic implementations" begin
        # Test that we're not missing any analytic implementation in `likelihoods_to_test`!
        default_quadrature_method_types = Set([
            m.sig.types[2] for m in methods(ApproximateGPs._default_quadrature)
        ])
        delete!(default_quadrature_method_types, Any)  # ignore fallback
        for lik in default_quadrature_method_types
            if nameof(lik) == :GammaLikelihood
                # workaround while waiting for JuliaGaussianProcesses/GPLikelihoods.jl#41
                @test any(nameof(lik) == nameof(l) for l in likelihoods_to_test)
            else
                @test any(lik <: l for l in likelihoods_to_test)
            end
        end
    end

    @testset "$lik" for lik in likelihoods_to_test
        l = lik()
        methods = [GaussHermite(100), MonteCarlo(1e7)]
        def = ApproximateGPs._default_quadrature(l)
        if def isa Analytic
            push!(methods, def)
        end
        y = rand.(rng, l.(zeros(10)))

        results = map(m -> ApproximateGPs.expected_loglik(m, y, q_f, l), methods)
        @test all(x -> isapprox(x, results[end]; atol=1e-6, rtol=1e-3), results)
    end
end
