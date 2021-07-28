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
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

k = [10, 0.1]

kernel = make_kernel(k)
f = LatentGP(GP(kernel), BernoulliLikelihood(), 0.1)
fx = f(x)


# %%
# Then, plot some samples from the prior underlying GP
x_plot = 0:0.02:6
prior_f_samples = rand(f.f(x_plot, 1e-6),20)

plt = plot(
    x_plot,
    prior_f_samples;
    seriescolor="red",
    linealpha=0.2,
    label=""
)
scatter!(plt, x, y; seriescolor="blue", label="Data points")


# %%
# Plot the same samples, but pushed through a logistic sigmoid to constrain
# them in (0, 1).
prior_y_samples = mean.(f.lik.(prior_f_samples))

plt = plot(
    x_plot,
    prior_y_samples;
    seriescolor="red",
    linealpha=0.2,
    label=""
)
scatter!(plt, x, y; seriescolor="blue", label="Data points")


# %%
# A simple Flux model
using Flux

struct SVGPModel
    k # kernel parameters
    m # variational mean
    A # variational covariance
    z # inducing points
end

@Flux.functor SVGPModel (k, m, A,) # Don't train the inducing inputs

lik = BernoulliLikelihood()
jitter = 1e-4

function (m::SVGPModel)(x)
    kernel = make_kernel(m.k)
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(m.m, m.A'm.A)
    fx = f(x)
    fu = f(m.z).fx
    return fx, fu, q
end

function flux_loss(x, y; n_data=length(y))
    fx, fu, q = model(x)
    return -SparseGPs.elbo(fx, y, fu, q; n_data, method=MonteCarlo())
end

# %%
M = 15 # number of inducing points

# Initialise the parameters
k = [10, 0.1]
m = zeros(M)
A = Matrix{Float64}(I, M, M)
z = x[1:M]

model = SVGPModel(k, m, A, z)

opt = ADAM(0.1)
parameters = Flux.params(model)

# %%
# Negative ELBO before training
println(flux_loss(x, y))

# %%
# Train the model
Flux.train!(
    (x, y) -> flux_loss(x, y),
    parameters,
    ncycle([(x, y)], 2000), # Train for 1000 epochs
    opt
)

# %%
# Negative ELBO after training
println(flux_loss(x, y))

# %%
# After optimisation, plot samples from the underlying posterior GP.
fu = f(z).fx # want the underlying FiniteGP
post = SparseGPs.approx_posterior(SVGP(), fu, MvNormal(m, A'A))
l_post = LatentGP(post, BernoulliLikelihood(), jitter)

post_f_samples = rand(l_post.f(x_plot, 1e-6), 20)

plt = plot(
    x_plot,
    post_f_samples;
    seriescolor="red",
    linealpha=0.2,
    legend=false
)

# %%
# As above, push these samples through a logistic sigmoid to get posterior predictions.
post_y_samples = mean.(l_post.lik.(post_f_samples))

plt = plot(
    x_plot,
    post_y_samples;
    seriescolor="red",
    linealpha=0.2,
    # legend=false,
    label=""
)
scatter!(plt, x, y; seriescolor="blue", label="Data points")
vline!(z; label="Pseudo-points")
