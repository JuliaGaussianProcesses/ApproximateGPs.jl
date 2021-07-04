@testset "equivalences" begin
    rng, N = MersenneTwister(654321), 20
    x = rand(rng, N)
    y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)

    z = copy(x) # Set inducing inputs == training inputs

    # Create a kernel from parameters k
    kernel(k) = softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
    k_init = [0.1, 0.1] # initial kernel parameters

    lik_noise = 0.1 # The (fixed) Gaussian likelihood noise

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

    # SGPR - Sparse GP regression (Titsias 2009)
    struct SGPRModel
        k # kernel parameters
        z # inducing points
    end
    @Flux.functor SGPRModel (k,) # Don't train the inducing inputs

    function (m::SGPRModel)(x)
        f = GP(kernel(m.k))
        fx = f(x, lik_noise)
        fz = f(m.z, lik_noise)
        return fx, fz
    end

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
        fz = f(m.z, lik_noise)
        return fx, fz, q
    end

    ## SECOND - create the models and associated training losses
    gpr = GPRModel(copy(k_init))
    function GPR_loss(x, y)
        fx = gpr(x)
        return -logpdf(fx, y)
    end

    sgpr = SGPRModel(copy(k_init), copy(z))
    function SGPR_loss(x, y)
        fx, fz = sgpr(x)
        return -AbstractGPs.elbo(fx, y, fz)
    end

    m, A = rand(rng, N), rand(rng, N, N) # initialise the variational parameters
    svgp = SVGPModel(copy(k_init), copy(z), m, A)
    function SVGP_loss(x, y)
        fx, fz, q = svgp(x)
        return -SparseGPs.elbo(fx, y, fz, q)
    end

    ## THIRD - train the models
    data = [(x, y)]
    opt = ADAM(0.01)

    Flux.train!((x, y) -> GPR_loss(x, y), Flux.params(gpr), ncycle(data, 300), opt)
    Flux.train!((x, y) -> SGPR_loss(x, y), Flux.params(sgpr), ncycle(data, 300), opt)
    Flux.train!((x, y) -> SVGP_loss(x, y), Flux.params(svgp), ncycle(data, 300), opt)

    ## FOURTH - test equivalence
    println(gpr.k)
    println(sgpr.k)
    println(svgp.k)
    @test gpr.k ≈ svgp.k

    # TODO: test posterior predictions
end

