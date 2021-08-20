# A recreation of <https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html>

# # Stochastic Variational Regression
#
# In this example, we show how to construct and train the stochastic variational
# Gaussian process (SVGP) model for efficient inference in large scale datasets.
# For a basic introduction to the functionality of this library, please refer to
# the [User Guide](@ref).
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
#md nothing #hide

# ## Generate some training data
#
# The data generating function

function g(x)
    return sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)
end

N = 10000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

scatter(x, y; xlabel="x", ylabel="y", markershape=:xcross, markeralpha=0.1, legend=false)

# ## Set up a Flux model
#
# We shall use the excellent framework provided by [Flux.jl](https://fluxml.ai/)
# to perform stochastic optimisation. The SVGP approximation has three sets of
# parameters to optimise - the inducing input locations, the mean and covariance
# of the variational distribution `q` and the parameters of the
# kernel.
#
# First, we define a helper function to construct the kernel from its parameters
# (often called kernel hyperparameters), and pick some initial values `k_init`.

using Flux

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

k_init = [0.3, 10]
#md nothing #hide

# Then, we select some inducing input locations `z_init`. In this case, we simply choose
# the first `M` data inputs.

M = 50 # number of inducing points
z_init = x[1:M]
#md nothing #hide

# Finally, we initialise the parameters of the variational distribution `q(u)`
# where `u ~ f(z)`. We parameterise the covariance matrix of `q` as `C = AᵀA`
# since this guarantees that `C` is positive definite.

m_init = zeros(M)
A_init = Matrix{Float64}(I, M, M)
q_init = MvNormal(m_init, A_init'A_init)
#md nothing #hide

# Given a set of parameters, we now define a Flux 'layer' which forms the basis
# of our model.

struct SVGPModel
    k  # kernel parameters
    m  # variational mean
    A  # variational covariance
    z  # inducing points
end

Flux.@functor SVGPModel (k, m, A, z)
#md nothing #hide

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
#md nothing #hide

# Create the posterior GP from the model parameters.
function posterior(m::SVGPModel)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    fu = f(m.z, jitter)
    q = MvNormal(m.m, m.A'm.A)
    return approx_posterior(SVGP(), fu, q)
end
#md nothing #hide

# Return the loss given data - in this case the negative ELBO.
function loss(x, y; n_data=length(y))
    fx, fu, q = model(x)
    return -elbo(fx, y, fu, q; n_data)
end
#md nothing #hide

model = SVGPModel(k_init, m, A, z_init)

opt = ADAM(0.001)
parameters = Flux.params(model)
#md nothing #hide

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
#md nothing #hide

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
vline!(z_init; label="Pseudo-points")
