# TODO: This doesn't actually use ApproximateGPs except for expected_loglik
#   what should be put in the library itself/what should the API be?
# TODO: Optimise z using VFE
# TODO: Improve numerical issues
# TODO: A better example - log Gaussian Cox process?

# # MCMC for Variationally Sparse Gaussian Processes
#
# This example is an implementation of [MCMC for Variationally Sparse Gaussian
# Processes](https://arxiv.org/abs/1506.04000). This approach allows for
# efficiently sampling from a non-Gaussian posterior approximation to both the
# kernel hyperparameters and latent function values in a GP model with any
# non-Gaussian likelihood ``p(y|f)``.

# ## Setup

using ApproximateGPs
using Plots
using LogExpFunctions
using LinearAlgebra
using ForwardDiff
using AdvancedHMC
using Distributions

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234);

# ## Generate some training data
#
# For this example, we will try to fit an exponential regression model. That is,
# there is assumed to be a latent function ``f = g(x)`` and observations are
# sampled from ``p(y| λ=exp(f)) = λe^{-λy}``.
# Here, we use ``g(x) = \sin^2(x)``

N = 30
x = rand(N) * 6 .- 3

f = sin.(x).^2
lik = ExponentialLikelihood()
y = rand(lik(f))

scatter(x, y)


# A function to create the GP kernel from parameters θ.
make_kernel(θ) = softplus(θ[1]) * with_lengthscale(Matern32Kernel(), softplus(θ[2]))

# The optimal approximating distribution is given (up to a normalising constant) by:
# ```math
# \log q (v, θ) \propto E_{p(f|u = Rv)}[\log p (y|f)] + \log p(v) + \log p(θ)
# ```
# So, we can sample from it exactly via HMC.

## TODO: explain whitened parameterisation u = Rv
## This function computes E_{p(f|u = Rv, θ)}[\log p (y|f)] given v and θ
function gp_expected_loglik(params)
    v = params[3:end]
    kernel = make_kernel(params[1:2])

    K_zz = kernelmatrix(kernel, z)
    K_zx = kernelmatrix(kernel, z, x)
    K_xx_diag = kernelmatrix_diag(kernel, x)

    # p(f_n | u=Rv, θ) = N(μ_n, γ_n)
    # A = R⁻¹K_zx
    # RRᵀ = K_zz
    # μ = Aᵀv
    # γ = diag(K_xx - AᵀA)
    R = cholesky(K_zz).L
    A = R \ K_zx
    μ = A'v
    γ = K_xx_diag .- ApproximateGPs.diag_At_A(A)

    p_f_given_u = Normal.(μ, γ)
    ##
    E_loglik = ApproximateGPs.expected_loglik(DefaultQuadrature(), y, p_f_given_u, lik)
    return E_loglik
end

# log p(θ) + log p(v)
logprior(params) = logpdf(MvNormal(length(params), 1), params)

# log q (v, θ)
logjoint(params) = gp_expected_loglik(params) + logprior(params)

# Select inducing variables on a grid.
n_inducing = 15
z = range(-2.8, 2.8, length=n_inducing)

# Run an HMC sampler
n_samples = 8_000
n_adapts = 2_000

samples, _ = begin
    n_params = n_inducing + 2
    metric = DiagEuclideanMetric(n_params)
    hamiltonian = Hamiltonian(metric, logjoint, ForwardDiff)

    initial_params = rand(n_params)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_params)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    sample(hamiltonian, proposal, initial_params, n_params, adaptor, n_adapts; progress=false)
end

# HMC samples `v`, so we need to obtain the unwhitened `u` samples.
x_grid = range(-3, 3; step=0.01)  # Grid of test values to use for plotting

function transform_samples(samples)
    θ_samples = [s[1:2] for s in samples]
    v_samples = [s[3:end] for s in samples]
    u_samples = Vector{Vector{Float64}}(undef, length(samples))

    rate_samples = Array{Float64}(undef, length(samples), length(x_grid), 5)

    for i in 1:length(samples)
        v = v_samples[i]
        θ = θ_samples[i]
        kernel = make_kernel(θ)
        R = cholesky(kernelmatrix(kernel, z)).L
        u = R * v
        u_samples[i] = u

        ## Take 5 samples from each posterior GP defined by (u, θ)
        ## Then, transform to get samples of the latent exponential rate parameter
        gp = GP(kernel)
        post = posterior(gp(z), u)
        post_samples = rand(post(x_grid, 1e-10), 5)
        rates = mean.(lik.(post_samples))
        rate_samples[i, :, :] = rates
    end
    return u_samples, v_samples, θ_samples, rate_samples
end

u_samples, v_samples, θ_samples, rate_samples = transform_samples(samples)

# Plot the mean predicted rate parameter, as well as the 5-95 percentile range.
scatter(x, y; label="Data")
vline!(z; label="Pseudo points")

quant_5 = mapslices(s -> quantile(vcat(s...), 0.05), rate_samples; dims=[1, 3])[:, :, 1]'
quant_95 = mapslices(s -> quantile(vcat(s...), 0.95), rate_samples; dims=[1, 3])[:, :, 1]'
plot!(x_grid, quant_5; label="5-95 percentile range", fillrange=quant_95, fillalpha=0.35, linealpha=0)

mean_rate = mean(rate_samples; dims=(1, 3))[1,:,1]
plot!(x_grid, mean_rate; label="Mean rate prediction")
