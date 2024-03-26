@testset "nearest_neighbors" begin
    x = [1., 2., 3.5, 4.2, 5.9, 8.]
    kern = SqExponentialKernel()
    fx = GP(kern)(x, 0.0);
    x2 = 1.0:0.1:8
    y = sin.(x)

    @testset "Using all neighbors is the same as the exact GP" begin
        opt_pred = mean_and_cov(posterior(NearestNeighbors(length(x) - 1),
            fx, y)(x2))
        pred = mean_and_cov(posterior(fx, y)(x2))
        for i in 1:2
            @test all(isapprox.(opt_pred[i], pred[i]; atol=1e-4))
        end
    end

    @testset "Using nearest neighbors approximates the exact GP" begin
        opt_pred = mean_and_cov(posterior(NearestNeighbors(3), fx, y)(x2))
        pred = mean_and_cov(posterior(fx, y)(x2))
        for i in 1:2
            @test all(isapprox.(opt_pred[i], pred[i]; atol=1e-1))
        end
    end
    
    @testset "Using nearest neighbors approximates the exact log likelihood" begin
        l1 = approx_lml(NearestNeighbors(3), fx, y)
        l2 = logpdf(fx, y)
        @test isapprox(l1, l2; atol=1e-2)
    end

    @testset "Zygote can take gradients of the logpdf" begin
        function objective(lengthscale::Float64)
            kern2 =  with_lengthscale(kern, lengthscale)
            fx = GP(kern2)(x, 0.0)
            return approx_lml(NearestNeighbors(3), fx, y)
        end
        lml, grads = Zygote.withgradient(objective, 1.0)
        
        @test approx_lml(NearestNeighbors(3), fx, y) ≈ lml
        @test all(abs.(grads) .> 0)
        
        # Calling back into AD seems to make this fail
        # kernA = with_lengthscale(kern, 1.0)
        # kernB = Tangent{typeof(kernA)}(kernel=NoTangent(),
        #   transform= Tangent{ScaleTransform{Float64}}(s = [FiniteDifferences.rand_tangent(1.0)],))
        # test_rrule(NearestNeighborsModule.make_B, x, 3, kernA ⊢ kernB)
        
        # This leads to cryptic FiniteDifferences errors
        # config = Zygote.ZygoteRuleConfig()
        # test_rrule(config, objective, 1.0;
            # rrule_f=rrule_via_ad, check_inferred=false)
    end
end
