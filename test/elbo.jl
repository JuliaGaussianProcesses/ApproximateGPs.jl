@testset "elbo" begin
    rng, N = MersenneTwister(654321), 20
    x = rand(rng, N) * 10
    y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)
    z = x[1:5]

    kernel = make_kernel([0.2, 0.6])
    f = GP(kernel)
    fx = f(x, 0.1)
    fz = f(z)
    q_ex = exact_variational_posterior(fz, fx, y)

    svgp = SVGP(fz, q_ex)
    @test elbo(svgp, fx, y) isa Real
    @test elbo(svgp, fx, y) ≤ logpdf(fx, y)

    fx_bad = f(x, fill(0.1, N))
    @test_throws ErrorException elbo(svgp, fx_bad, y)

    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))

    analytic_likelihoods = [
        ExponentialLikelihood, GammaLikelihood, PoissonLikelihood, GaussianLikelihood
    ]

    @testset "testing all analytic implementations" begin
        ### Test that we're not missing any analytic implementation in `analytic_likelihoods`!
        default_quadrature_method_types = Set([
            m.sig.types[2] for m in methods(ApproximateGPs._default_quadrature)
        ])
        delete!(default_quadrature_method_types, Any)  # ignore fallback
        for lik in default_quadrature_method_types
            if lik.body.name.name == :GammaLikelihood
                # workaround while waiting for JuliaGaussianProcesses/GPLikelihoods.jl#41
                @test any(
                    l.body.name.name == lik.body.name.name for l in analytic_likelihoods
                )
            else
                @test any(lik <: l for l in analytic_likelihoods)
            end
        end
    end

    @testset "$lik" for lik in analytic_likelihoods
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
