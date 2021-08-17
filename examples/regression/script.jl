# A recreation of https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

# # One-Dimensional Stochastic Variational Regression
#
# ## Setup

using AbstractGPs
using SparseGPs
using Distributions
using LinearAlgebra
using IterTools

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)

# ## Generate some training data
#
# The data generating function

function g(x)
    return sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)
end

N = 10000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

scatter(x, y; xlabel="x", ylabel="y", legend=false)

# ## Set up a Flux SVGP model
#
# First, a helper function to create the GP kernel

using Flux

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

struct SVGPModel
    k  # kernel parameters
    m  # variational mean
    A  # variational covariance
    z  # inducing points
end

Flux.@functor SVGPModel (k, m, A) # Don't train the inducing inputs

# Create the 'model' from the parameters - i.e. return the FiniteGP at inputs x,
# the FiniteGP at inducing inputs z and the variational posterior over inducing
# points - q(u).

lik_noise = 0.3
jitter = 1e-5

function (m::SVGPModel)(x)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    q = MvNormal(m.m, m.A'm.A)
    fx = f(x, lik_noise)
    fu = f(m.z, jitter)
    return fx, fu, q
end

# Create the posterior GP from the model parameters.
function posterior(m::SVGPModel)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    fu = f(m.z, jitter)
    q = MvNormal(m.m, m.A'm.A)
    return approx_posterior(SVGP(), fu, q)
end

# Return the loss given data - in this case the negative ELBO.
function loss(x, y; n_data=length(y))
    fx, fu, q = model(x)
    return -elbo(fx, y, fu, q; n_data)
end
#md nothing #hide

# Select the first M inputs as inducing inputs

M = 50 # number of inducing points
z = x[1:M]

# Initialise the model parameters

k = [0.3, 10]
m = zeros(M)
A = Matrix{Float64}(I, M, M)

model = SVGPModel(k, m, A, z)

opt = ADAM(0.001)
parameters = Flux.params(model)

# TODO: batching explanation

b = 100 # minibatch size
data_loader = Flux.Data.DataLoader((x, y); batchsize=b)

# The loss (negative ELBO) before training

println(loss(x, y))

# Train the model

Flux.train!(
    (x, y) -> loss(x, y; n_data=N),
    parameters,
    ncycle(data_loader, 300), # Train for 300 epochs
    opt,
)

# Negative ELBO after training

println(loss(x, y))

# Plot samples from the optmimised approximate posterior.

post = posterior(model)

scatter(
    x,
    y;
    markershape=:xcross,
    markeralpha=0.1,
    xlim=(-1, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (VI with sparse grid)",
    label="Train Data",
)
plot!(-1:0.001:1, post; label="Posterior")
plot!(-1:0.001:1, g; label="True Function")
vline!(z; label="Pseudo-points")
