# (Roughly) a recreation of https://gpflow.readthedocs.io/en/master/notebooks/basics/classification.html

# # One-Dimensional Stochastic Variational Classification
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

# First, a helper function to create the GP kernel

function make_kernel(k)
    return softplus(k[1]) * (Matern52Kernel() ∘ ScaleTransform(softplus(k[2])))
end
kernel_params = [10.0, 0.5] # The true kernel parameters
#md nothing #hide

# Create the 'ground truth GP' and sample a function

k = make_kernel(kernel_params)
f = LatentGP(GP(k), BernoulliLikelihood(), 1e-6)
x_true = 0:0.02:6
f_true, y_true = rand(f(x_true))
#md nothing #hide

# Plot the sampled function
plot(x_true, f_true; seriescolor="red", label="")

# Plot the function pushed through a logistic sigmoid, restricting it to ``[0, 1]``
plot(x_true, mean.(f.lik.(f_true)); seriescolor="red", label="")

# Subsample input locations to obtain training data

N_train = 30
mask = sample(1:length(x_true), N_train; replace=false, ordered=true)
x, _, y = x_true[mask], f_true[mask], y_true[mask]
scatter(x, y; label="Sampled data")
plot!(x_true, mean.(f.lik.(f_true)); seriescolor="red", label="True function")

# ## Setting up the SVGP model

M = 15 # number of inducing points
z = sample(x, M; replace=false)

model = SVGPModel(
    make_kernel,    # The function to create the kernel
    rand(2),        # The initial kernel parameters
    z;              # The inducing inputs
    jitter=1e-4,
    likelihood=BernoulliLikelihood()
)
#md nothing #hide

# Plot some samples from the prior of this model

f = prior(model)
fx = f(x)
x_plot = 0:0.02:6
prior_f_samples = rand(f.f(x_plot), 20)

plot(x_plot, prior_f_samples; seriescolor="red", linealpha=0.2, label="")

# Plot the same samples, but pushed through a logistic sigmoid as before

prior_y_samples = mean.(f.lik.(prior_f_samples))

plot(x_plot, prior_y_samples; seriescolor="red", linealpha=0.2, label="")
scatter!(x, y; seriescolor="blue", label="Data points")

# ## Train the model using Flux

opt = ADAM(0.1)
parameters = Flux.params(model)
delete!(parameters, model.z)    # Don't train the inducing inputs
#md nothing #hide

# Negative ELBO before training

println(loss(model, x, y))

# The main training loop

Flux.train!(
    (x, y) -> loss(model, x, y),
    parameters,
    ncycle([(x, y)], 6000), # Train for 6000 epochs
    opt,
)
#md nothing #hide

# Negative ELBO after training

println(loss(model, x, y))

# After optimisation, plot samples from the underlying posterior GP.

post = posterior(model)
post_f_samples = rand(post.f(x_plot), 20)

plot(x_plot, post_f_samples; seriescolor="red", linealpha=0.2, legend=false)

# As above, push these samples through a logistic sigmoid to get posterior samples.

post_y_samples = mean.(post.lik.(post_f_samples))

plot(
    x_plot,
    post_y_samples;
    seriescolor="green",
    linealpha=0.2,
    label="",
)
scatter!(x, y; seriescolor="blue", label="Data points")
vline!(z; label="Pseudo-points")
plot!(x_true, mean.(f.lik.(f_true)); seriescolor="red", linewidth=3, label="True function")
