# (Roughly) a recreation of https://gpflow.readthedocs.io/en/master/notebooks/basics/classification.html

# # Stochastic Variational Classification
#
# ## Setup

using SparseGPs
using AbstractGPs
using GPLikelihoods
using Distributions
using LinearAlgebra
using IterTools
using Flux

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)

# ## Sampling some data
#
# In this example, we shall see whether the sparse variational Gaussian process
# (SVGP) can recover the true GP from which binary classification data are
# sampled.
#
# First, a helper function to create the GP kernel

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end
kernel_params = [10.0, 0.5] # The true kernel parameters
k = make_kernel(kernel_params)
#md nothing #hide

# Create the 'ground truth GP' and sample a function

f = LatentGP(GP(k), BernoulliLikelihood(), 1e-6)
x_true = 0:0.02:6
f_true, y_true = rand(f(x_true))
#md nothing #hide

# Plot the sampled function
plot(x_true, f_true; seriescolor="red", label="")

# Plot the function pushed through a logistic sigmoid, restricting it to `[0, 1]`
plot(x_true, mean.(f.lik.(f_true)); seriescolor="red", label="")

# Subsample input locations to obtain training data

N_train = 30
mask = sample(1:length(x_true), N_train; replace=false, ordered=true)
x, y = x_true[mask], y_true[mask]
scatter(x, y; label="Sampled data")
plot!(x_true, mean.(f.lik.(f_true)); seriescolor="red", label="True function")

# ## Setting up a Flux model for the SVGP

struct SVGPModel
    k  # kernel parameters
    z  # inducing points
    m  # variational mean
    A  # variational covariance
end

Flux.@functor SVGPModel (k, m, A)  # Don't train the inducing inputs

lik = BernoulliLikelihood()
jitter = 1e-3

function (m::SVGPModel)(x)
    kernel = make_kernel(m.k)
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(m.m, m.A'm.A)
    fx = f(x)
    fu = f(m.z).fx
    return fx, fu, q
end

function loss(x, y; n_data=length(y))
    fx, fu, q = model(x)
    return -elbo(fx, y, fu, q; n_data, method=MonteCarlo())
end
#md nothing #hide

# Initialise the model parameters

M = 15  # number of inducing points
k = rand(2)
m = zeros(M)
A = Matrix{Float64}(I, M, M)
z = range(0; stop=6, length=M)

model = SVGPModel(k, m, A, z)

opt = ADAM(0.1)
parameters = Flux.params(model)
#md nothing #hide

# The loss (negative ELBO) before training

println(loss(x, y))

# Train the model

Flux.train!(
    (x, y) -> loss(x, y),
    parameters,
    ncycle([(x, y)], 6000), # Train for 6000 epochs
    opt,
)

# The loss after training

println(loss(x, y))

# After optimisation, plot samples from the underlying posterior GP.

fu = f(z).fx # want the underlying FiniteGP
post = approx_posterior(SVGP(), fu, MvNormal(m, A'A))
l_post = LatentGP(post, BernoulliLikelihood(), jitter)

x_plot = 0:0.02:6

post_f_samples = rand(l_post.f(x_plot, 1e-6), 20)

plt = plot(x_plot, post_f_samples; seriescolor="red", linealpha=0.2, legend=false)

# As above, push these samples through a logistic sigmoid to get posterior predictions.

post_y_samples = mean.(l_post.lik.(post_f_samples))

plt = plot(x_plot, post_y_samples; seriescolor="red", linealpha=0.2, label="")
scatter!(plt, x, y; seriescolor="blue", label="Data points")
vline!(z; label="Pseudo-points")
plot!(
    x_true, mean.(f.lik.(f_true)); seriescolor="green", linewidth=3, label="True function"
)
