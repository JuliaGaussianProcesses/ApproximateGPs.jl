@testset "equivalences" begin
    rng, N = MersenneTwister(654321), 20
    x = rand(rng, N) * 10
    y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)

    z = copy(x) # Set inducing inputs == training inputs

    # Create a kernel from parameters k
    kernel(k) = softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
    k_init = [0.2, 0.6] # initial kernel parameters

    lik_noise = 0.1 # The (fixed) Gaussian likelihood noise
    jitter = 1e-5

    ## FIRST - define the models
    # GPR - Exact GP regression
    struct GPRModel
        k # kernel parameters
    end
    @Flux.functor GPRModel

    function (m::GPRModel)(x)
        f = GP(kernel(m.k))
        fx = f(x, lik_noise)
        return fx
    end

    # # SGPR - Sparse GP regression (Titsias 2009)
    # struct SGPRModel
    #     k # kernel parameters
    #     z # inducing points
    # end
    # @Flux.functor SGPRModel (k,) # Don't train the inducing inputs

    # function (m::SGPRModel)(x)
    #     f = GP(kernel(m.k))
    #     fx = f(x, lik_noise)
    #     fz = f(m.z, lik_noise)
    #     return fx, fz
    # end

    # SVGP - Sparse variational GP regression (Hensman 2014)
    struct SVGPModel
        k # kernel parameters
        z # inducing points
        m # variational mean
        A # variational covariance sqrt (Σ = A'A)
    end
    @Flux.functor SVGPModel (k, m, A,) # Don't train the inducing inputs

    function (m::SVGPModel)(x)
        f = GP(kernel(m.k))
        q = MvNormal(m.m, m.A'm.A)
        fx = f(x, lik_noise)
        fz = f(m.z, jitter)
        return fx, fz, q
    end

    ## SECOND - create the models and associated training losses
    gpr = GPRModel(copy(k_init))
    function GPR_loss(x, y)
        fx = gpr(x)
        return -logpdf(fx, y)
    end

    # sgpr = SGPRModel(copy(k_init), copy(z))
    # function SGPR_loss(x, y)
    #     fx, fz = sgpr(x)
    #     return -AbstractGPs.elbo(fx, y, fz)
    # end

    m, A = rand(rng, N), rand(rng, N, N)/2 # initialise the variational parameters
    svgp = SVGPModel(copy(k_init), copy(z), m, A)
    function SVGP_loss(x, y)
        fx, fz, q = svgp(x)
        return -SparseGPs.elbo(fx, y, fz, q)
    end

    ## THIRD - train the models
    data = [(x, y)]
    opt = ADAM(0.01)

    svgp_ps = Flux.params(svgp)
    delete!(svgp_ps, svgp.k) # Don't train the kernel parameters

    # Flux.train!((x, y) -> GPR_loss(x, y), Flux.params(gpr), ncycle(data, 3000), opt)
    # Flux.train!((x, y) -> SGPR_loss(x, y), Flux.params(sgpr), ncycle(data, 3000), opt)
    Flux.train!((x, y) -> SVGP_loss(x, y), svgp_ps, ncycle(data, 9000), opt)

    ## FOURTH - construct the posteriors
    function posterior(m::GPRModel, x, y)
        f = GP(kernel(m.k))
        fx = f(x, lik_noise)
        return AbstractGPs.posterior(fx, y)
    end

    # function posterior(m::SGPRModel, x, y)
    #     f = GP(kernel(m.k))
    #     fx = f(x, lik_noise)
    #     fz = f(m.z)
    #     return AbstractGPs.approx_posterior(VFE(), fx, y, fz)
    # end

    function posterior(m::SVGPModel)
        f = GP(kernel(m.k))
        fz = f(m.z, jitter)
        q = MvNormal(m.m, m.A'm.A)
        return SparseGPs.approx_posterior(SVGP(), fz, q)
    end
    gpr_post = posterior(gpr, x, y)
    # sgpr_post = posterior(sgpr, x, y)
    svgp_post = posterior(svgp)

    ## FIFTH - test equivalences
    @test all(isapprox.(mean(gpr_post, x), mean(svgp_post, x), atol=1e-3))
    @test all(isapprox.(cov(gpr_post, x), cov(svgp_post, x), atol=1e-3))

end

