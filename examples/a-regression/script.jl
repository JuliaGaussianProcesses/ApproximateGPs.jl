# A recreation of <https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html>

# # Regression: Sparse Variational Gaussian Process for Stochastic Optimisation with Flux.jl

# In this example, we show how to construct and train the stochastic variational
# Gaussian process (SVGP) model for efficient inference in large scale datasets.
# For a basic introduction to the functionality of this library, please refer to
# the [User Guide](@ref).
#
# ## Setup

using ApproximateGPs
using Distributions
using LinearAlgebra
using DisplayAs #hide

using Plots
default(; fmt=:png, palette=:seaborn_colorblind, legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234);

# ## Generate some training data
#
# We define a data-generating function `g`:

g(x) = sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)

N = 10000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

plt = scatter(
    x, y; xlabel="x", ylabel="y", markershape=:xcross, markeralpha=0.1, legend=false
)
DisplayAs.PNG(plt) #hide

# ## Set up a Flux model
#
# We shall use the excellent framework provided by [Flux.jl](https://fluxml.ai/)
# to perform stochastic optimisation. The SVGP approximation has three sets of
# parameters to optimise - the inducing input locations, the mean and covariance
# of the variational distribution `q` and the parameters of the
# kernel.
#
# First, we define a helper function to construct the kernel from its
# parameters (often called kernel hyperparameters), and pick some
# initial values `k_init`. Since Flux performs optimisation on
# unconstrained parameters, we need to use softplus to ensure that the
# kernel parameters are positive.

using StatsFuns: softplus, invsoftplus

function make_kernel(k_params)
    variance = softplus(k_params[1])
    lengthscale = softplus(k_params[2])
    return variance * with_lengthscale(SqExponentialKernel(), lengthscale)
end

init_variance = 1.3
init_lengthscale = 0.3
k_init = [invsoftplus(init_variance), invsoftplus(init_lengthscale)];

# Then, we select some inducing input locations `z_init`. In this
# case, we simply choose the first `M` data inputs.

M = 20 # number of inducing points
z_init = x[1:M];

# Given a set of parameters, we now define a Flux 'layer' which forms
# the basis of our model.

using Flux

struct SVGPModel
    k  # kernel parameters
    z  # inducing points
    m  # variational mean
    A  # square-root of variational covariance
end

Flux.@functor SVGPModel (k, z, m, A);

# Set the observation noise for our model, along with a `jitter` term
# to help with numerical stability.

lik_noise = 0.3
jitter = 1e-5;

# Next, we define some useful functions on the model - creating the prior GP
# under the model, as well as the `SparseVariationalApproximation` struct
# needed to create the posterior approximation and to compute the ELBO.

function prior(m::SVGPModel)
    kernel = make_kernel(m.k)
    return GP(kernel)
end

# The variational distribution is given by `q(u)` where ``u ~ f(z)`` are the
# pseudo-points. We parameterise the covariance matrix of ``q`` as ``S = A A^T``
# since this guarantees that ``S`` is positive definite. We also only use the
# lower triangular part of `A`, to ensure the minimum number of free parameters.

using PDMats: PDMat

function make_approx(m::SVGPModel, prior)
    ## Efficiently constructs S as A*Aᵀ
    S = PDMat(Cholesky(LowerTriangular(m.A)))
    q = MvNormal(m.m, S)
    fz = prior(m.z, jitter)
    return SparseVariationalApproximation(fz, q)
end;

# Create the approximate posterior GP under the model.

function model_posterior(m::SVGPModel)
    svgp = make_approx(m, prior(m))
    return posterior(svgp)
end;

# Define a predictive function for the model - in this case the prediction is
# the joint distribution of the approximate posterior GP at some test inputs `x`
# (defined by an `AbstractGPs.FiniteGP`).

function (m::SVGPModel)(x)
    post = model_posterior(m)
    return post(x)
end;

# Return the loss given data - for the SVGP model as constructed using `SparseVariationalApproximation`, the
# loss used is the negative ELBO (also known as the Variational Free Energy).
# `num_data` is required for minibatching used below.

function loss(m::SVGPModel, x, y; num_data=length(y))
    f = prior(m)
    fx = f(x, lik_noise)
    svgp = make_approx(m, f)
    return -elbo(svgp, fx, y; num_data)
end;

# Finally, we choose some initial parameters and instantiate our model.

m_init = zeros(M)
A_init = Matrix{Float64}(I, M, M)

model = SVGPModel(k_init, z_init, m_init, A_init);

# Taking a look at the model posterior under these initial parameters shows a
# very poor fit to the data, as expected:

init_post = model_posterior(model)
plt = scatter(
    x,
    y;
    xlabel="x",
    ylabel="y",
    markershape=:xcross,
    markeralpha=0.1,
    label="Training Data",
)
plot!(plt, -1:0.001:1, init_post; label="Initial Posterior", color=4)
DisplayAs.PNG(plt) #hide

# ## Training the model
#
# Training the model now simply proceeds with the usual `Flux.jl` training loop.

opt = ADAM(0.001)  # Define the optimiser
params = Flux.params(model);  # Extract the model parameters

# One of the major advantages of the SVGP model is that it allows stochastic
# estimation of the ELBO by using minibatching of the training data. This is
# very straightforward to achieve with `Flux.jl`'s utilities:

b = 100 # minibatch size
data_loader = Flux.Data.DataLoader((x, y); batchsize=b)

# The loss (negative ELBO) before training

loss(model, x, y)

# Train the model. N.B. when using minibatching, the length of the
# full dataset `num_data` must be passed to the loss.

using IterTools: ncycle

Flux.train!(
    (x, y) -> loss(model, x, y; num_data=N),
    params,
    ncycle(data_loader, 300), # Train for 300 epochs
    opt,
);

# Negative ELBO after training

loss(model, x, y)

# Finally, we plot the optimised approximate posterior to see the
# results.

post = model_posterior(model)

plt = scatter(
    x,
    y;
    markershape=:xcross,
    markeralpha=0.1,
    xlim=(-1, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (VI with sparse grid)",
    label="Training Data",
    color=1,
)
plot!(plt, -1:0.001:1, post; label="Posterior", color=4)
sticks!(plt, model.z, fill(0.13, M); label="Pseudo-points", linewidth=1.5, color=5)
DisplayAs.PNG(plt) #hide
