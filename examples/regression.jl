# A recreation of https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

# # One-Dimensional Stochastic Variational Regression
#
# ## Setup

using AbstractGPs
using SparseGPs
using Distributions
using LinearAlgebra
using Optim
using IterTools
using GPLikelihoods
using Flux

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

# ## Set up the SVGP model
#
# First, a helper function to create the GP kernel

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

# Select the first M inputs as inducing inputs and initialise the kernel parameters

M = 50
z = x[1:M]
k_init = [0.3, 10]

# Create the model and optimiser

model = SVGPModel(make_kernel, k_init, z; likelihood=GaussianLikelihood(0.1))

opt = ADAM(0.001)
parameters = Flux.params(model)
delete!(parameters, model.z)    # Don't train the inducing inputs
#md nothing #hide

# To speed up training, we can estimate the loss function by using minibatching

b = 100 # minibatch size
data_loader = Flux.Data.DataLoader((x, y); batchsize=b)
#md nothing #hide

# Negative ELBO before training

println(loss(model, x, y))

# The main training loop

Flux.train!(
    (x, y) -> loss(model, x, y; n_data=N),
    parameters,
    ncycle(data_loader, 300), # Train for 300 epochs
    opt,
)
#md nothing #hide

# Negative ELBO after training

println(loss(model, x, y))

# Plot samples from the optimised approximate posterior.

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
plot!(-1:0.001:1, post.f; label="Posterior")
vline!(z; label="Pseudo-points")
