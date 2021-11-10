@testset "sparse_variational" begin
    @testset "AbstractGPs interface" begin
        rng = MersenneTwister(123456)
        N_cond = 5
        N_a = 6
        N_b = 7

        # Specify prior.
        f = GP(Matern32Kernel())
        # Sample from prior.
        x = collect(range(-1.0, 1.0; length=N_cond))
        fx = f(x, 1e-15)
        y = rand(rng, fx)

        q = exact_variational_posterior(fx, fx, y)
        f_approx_post = posterior(SparseVariationalApproximation(fx, q))

        a = collect(range(-1.0, 1.0; length=N_a))
        b = randn(rng, N_b)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f_approx_post, a, b)
    end

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

        sva = SparseVariationalApproximation(fz, q_ex)
        @test elbo(sva, fx, y) isa Real
        @test elbo(sva, fx, y) ≤ logpdf(fx, y)

        fx_bad = f(x, fill(0.1, N))
        @test_throws ErrorException elbo(sva, fx_bad, y)

        lf = LatentGP(f, GaussianLikelihood(0.1), 1e-18)
        lfx = lf(x)

        @test elbo(sva, lfx, y) ≈ elbo(sva, fx, y) atol=1e-10
    end

    @testset "equivalences" begin
        rng, N = MersenneTwister(654321), 20
        x = rand(rng, N) * 10
        y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)

        z = copy(x) # Set inducing inputs == training inputs

        k_init = [0.2, 0.6] # initial kernel parameters
        lik_noise = 0.1 # The (fixed) Gaussian likelihood noise

        @testset "exact posterior" begin
            # There is a closed form optimal solution for the variational posterior
            # q(u) (e.g. # https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
            # equations (11) & (12)). The SVGP posterior with this optimal q(u)
            # should therefore be equivalent to the sparse GP (Titsias) posterior
            # and exact GP regression (when z == x).

            kernel = make_kernel(k_init)
            f = GP(kernel)
            fx = f(x, lik_noise)
            fz = f(z)
            q_ex = exact_variational_posterior(fz, fx, y)

            gpr_post = posterior(fx, y) # Exact GP regression
            vfe_post = posterior(VFE(fz), fx, y) # Titsias posterior
            svgp_post = posterior(SparseVariationalApproximation(fz, q_ex)) # Hensman (2013) exact posterior

            @test mean(gpr_post, x) ≈ mean(svgp_post, x) atol = 1e-10
            @test cov(gpr_post, x) ≈ cov(svgp_post, x) atol = 1e-10

            @test mean(vfe_post, x) ≈ mean(svgp_post, x) atol = 1e-10
            @test cov(vfe_post, x) ≈ cov(svgp_post, x) atol = 1e-10

            @test elbo(SparseVariationalApproximation(fz, q_ex), fx, y) ≈ logpdf(fx, y) atol =
                1e-6
        end

        @testset "optimised posterior" begin
            jitter = 1e-5

            make_gp(kernel) = GP(kernel)

            # SVGP model
            struct SVGPModel
                k # kernel parameters
                z # inducing points
                m # variational mean
                A # variational covariance sqrt (Σ = A'A)
            end
            Flux.@functor SVGPModel (m, A) # Only train the variational parameters

            function construct_parts(m::SVGPModel, x)
                f = make_gp(make_kernel(m.k))
                fx = f(x, lik_noise)
                fz = f(m.z, jitter)

                S = PDMat(Cholesky(LowerTriangular(m.A)))
                q = MvNormal(m.m, S)
                return SparseVariationalApproximation(fz, q), fx
            end

            m, A = zeros(N), Matrix{Float64}(I, N, N) # initialise the variational parameters
            svgp_model = SVGPModel(copy(k_init), copy(z), m, A)
            function svgp_loss(x, y)
                approx, fx = construct_parts(svgp_model, x)
                return -elbo(approx, fx, y)
            end

            # Train the SVGP model
            data = [(x, y)]
            opt = ADAM(0.001)

            svgp_ps = Flux.params(svgp_model)

            # Optimise q(u)
            Flux.train!((x, y) -> svgp_loss(x, y), svgp_ps, ncycle(data, 20000), opt)

            ## construct the posteriors
            f_gpr = make_gp(make_kernel(k_init))
            gpr_post = posterior(f_gpr(x, lik_noise), y)

            svgp_post = posterior(first(construct_parts(svgp_model, x)))

            ## FIFTH - test equivalences
            @test all(isapprox.(mean(gpr_post, x), mean(svgp_post, x), atol=1e-4))
            @test all(isapprox.(cov(gpr_post, x), cov(svgp_post, x), atol=1e-4))
        end
    end
end
