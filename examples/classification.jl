# Recreation of https://gpflow.readthedocs.io/en/master/notebooks/basics/classification.html

# %%
using SparseGPs
using AbstractGPs
using GPLikelihoods
using StatsFuns
using FastGaussQuadrature
using Distributions
using LinearAlgebra
using DelimitedFiles
using IterTools

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)

# %%
# Read in the classification data
data_file = pkgdir(SparseGPs) * "/examples/data/classif_1D.csv"
x, y = eachcol(readdlm(data_file))
scatter(x, y)

# %%
# First, create the GP kernel from given parameters k
function make_kernel(k)
    return softplus(k[1]) * (Matern52Kernel() ∘ ScaleTransform(softplus(k[2])))
end

k = [20.0, 0.5]
M = 15 # number of inducing points
z = x[1:M]

model = SVGPModel(make_kernel, k, z; jitter=1e-3, likelihood=BernoulliLikelihood())

f = prior(model)
fx = f(x)

# %%
# Then, plot some samples from the prior underlying GP
x_plot = 0:0.02:6
prior_f_samples = rand(f.f(x_plot), 20) # TODO: add jitter?

plt = plot(x_plot, prior_f_samples; seriescolor="red", linealpha=0.2, label="")
scatter!(plt, x, y; seriescolor="blue", label="Data points")

# %%
# Plot the same samples, but pushed through a logistic sigmoid to constrain
# them in (0, 1).
prior_y_samples = mean.(f.lik.(prior_f_samples))

plt = plot(x_plot, prior_y_samples; seriescolor="red", linealpha=0.2, label="")
scatter!(plt, x, y; seriescolor="blue", label="Data points")

# %%
# Optimise the model using Flux
using Flux

opt = ADAM(0.1)
parameters = Flux.params(model)
delete!(parameters, model.z)    # Don't train the inducing inputs

# %%
# Negative ELBO before training
println(loss(model, x, y))

# %%
# Train the model
Flux.train!(
    (x, y) -> loss(model, x, y),
    parameters,
    ncycle([(x, y)], 2000), # Train for 1000 epochs
    opt,
)

# %%
# Negative ELBO after training
println(loss(model, x, y))

# %%
# After optimisation, plot samples from the underlying posterior GP.
post = posterior(model)

post_f_samples = rand(post.f(x_plot), 20) # TODO: add jitter?

plt = plot(x_plot, post_f_samples; seriescolor="red", linealpha=0.2, legend=false)

# %%
# As above, push these samples through a logistic sigmoid to get posterior predictions.
post_y_samples = mean.(post.lik.(post_f_samples))

plt = plot(
    x_plot,
    post_y_samples;
    seriescolor="red",
    linealpha=0.2,
    # legend=false,
    label="",
)
scatter!(plt, x, y; seriescolor="blue", label="Data points")
vline!(z; label="Pseudo-points")
