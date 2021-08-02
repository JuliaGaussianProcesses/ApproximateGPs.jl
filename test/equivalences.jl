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
        vfe_post = approx_posterior(VFE(), fx, y, fz) # Titsias posterior
        svgp_post = approx_posterior(SVGP(), fz, q_ex) # Hensman (2013) exact posterior

        @test mean(gpr_post, x) ≈ mean(svgp_post, x) atol = 1e-10
        @test cov(gpr_post, x) ≈ cov(svgp_post, x) atol = 1e-10

        @test mean(vfe_post, x) ≈ mean(svgp_post, x) atol = 1e-10
        @test cov(vfe_post, x) ≈ cov(svgp_post, x) atol = 1e-10

        @test elbo(fx, y, fz, q_ex) ≈ logpdf(fx, y)
    end

    @testset "optimised posterior" begin
        jitter = 1e-5

        make_gp(kernel) = GP(kernel)

        ## FIRST - define the models
        # GPR - Exact GP regression
        struct GPRModel
            k # kernel parameters
        end
        Flux.@functor GPRModel

        function (m::GPRModel)(x)
            f = make_gp(make_kernel(m.k))
            fx = f(x, lik_noise)
            return fx
        end

        ## SECOND - create the models and associated training losses
        gpr = GPRModel(copy(k_init))
        function SparseGPs.loss(gpr::GPRModel, x, y)
            fx = gpr(x)
            return -logpdf(fx, y)
        end

        svgp = SVGPModel(make_kernel, copy(k_init), copy(z); jitter=jitter, likelihood=GaussianLikelihood(lik_noise))

        ## THIRD - train the models
        data = [(x, y)]
        opt = ADAM(0.001)

        svgp_ps = Flux.params(svgp)
        delete!(svgp_ps, svgp.k) # Don't train the kernel parameters
        delete!(svgp_ps, svgp.z) # Don't train the inducing points

        # Optimise q(u)
        Flux.train!((x, y) -> loss(svgp, x, y), svgp_ps, ncycle(data, 20000), opt)

        ## FOURTH - construct the posteriors
        function AbstractGPs.posterior(m::GPRModel, x, y)
            f = make_gp(make_kernel(m.k))
            fx = f(x, lik_noise)
            return AbstractGPs.posterior(fx, y)
        end

        gpr_post = posterior(gpr, x, y)
        svgp_post = posterior(svgp)

        ## FIFTH - test equivalences
        @test all(isapprox.(mean(gpr_post, x), mean(svgp_post, x), atol=1e-4))
        @test all(isapprox.(cov(gpr_post, x), cov(svgp_post, x), atol=1e-4))
    end
end
