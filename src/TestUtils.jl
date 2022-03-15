module TestUtils

using LinearAlgebra
using Random
using Test

using Distributions
using LogExpFunctions: logistic

using AbstractGPs
using ApproximateGPs

function generate_data()
    X = range(0, 23.5; length=48)
    # The random number generator changed in 1.6->1.7. The following vector was generated in Julia 1.6.
    # The generating code below is only kept for illustrative purposes.
    #! format: off
    Y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    #! format: on
    # Random.seed!(1)
    # fs = @. 3 * sin(10 + 0.6X) + sin(0.1X) - 1
    # # invlink = normcdf
    # invlink = logistic
    # ps = invlink.(fs)
    # Y = @. rand(Bernoulli(ps))
    return X, Y
end

dist_y_given_f(f) = Bernoulli(logistic(f))

function build_latent_gp(theta)
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return LatentGP(GP(kernel), dist_y_given_f, 1e-8)
end

function test_approximation_predictions(approx)
    rng = MersenneTwister(123456)
    N_cond = 5
    N_a = 6
    N_b = 7

    # Specify prior.
    f = GP(Matern32Kernel())
    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    noise_scale = 0.1
    fx = f(x, noise_scale^2)
    y = rand(rng, fx)

    jitter = 0.0  # not needed in Gaussian case
    lf = LatentGP(f, f -> Normal(f, noise_scale), jitter)
    f_approx_post = posterior(approx, lf(x), y)

    @testset "AbstractGPs API" begin
        a = collect(range(-1.2, 1.2; length=N_a))
        b = randn(rng, N_b)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f_approx_post, a, b)
    end

    @testset "exact GPR equivalence for Gaussian likelihood" begin
        f_exact_post = posterior(f(x, noise_scale^2), y)
        xt = vcat(x, randn(rng, 3))  # test at training and new points

        m_approx, c_approx = mean_and_cov(f_approx_post(xt))
        m_exact, c_exact = mean_and_cov(f_exact_post(xt))

        @test m_approx ≈ m_exact
        @test c_approx ≈ c_exact
    end
end

end
