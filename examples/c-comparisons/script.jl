# # Binary Classification with Various Approximations
# 
#
# This example demonstrates how to carry out non-conjugate Gaussian process
# inference using different approximations.
# For a basic introduction to the functionality of this library, please refer to the
# [User Guide](@ref).
#
# ## Setup

using ApproximateGPs
using LinearAlgebra
using Distributions
using LogExpFunctions: logistic
#using ParameterHandling
using Zygote
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1);

# ## Generate some training data

Xgrid = -4:0.1:29
X = range(0, 23.5; length=48)
f(x) = 3 * sin(10 + 0.6x) + sin(0.1x) - 1
fs = f.(X)
invlink = logistic  # could use other invlink, e.g. normcdf(f) = cdf(Normal(), f)
ps = invlink.(fs)
Y = [rand(Bernoulli(p)) for p in ps]

function plot_data()
    plot(; xlims=extrema(Xgrid), xticks=0:6:24)
    plot!(Xgrid, invlink ∘ f; label="true probabilities")
    return scatter!(X, Y; label="observations", color=3)
end

plot_data()

# ## Creating the latent GP

dist_y_given_f(f) = Bernoulli(invlink(f))

function build_latent_gp(theta)
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return LatentGP(GP(kernel), dist_y_given_f, 1e-8)
end

function plot_samples!(Xgrid, fpost; samples=100, color=2)
    fx = fpost(Xgrid, 1e-8)
    fsamples = rand(fx, samples)
    plot!(Xgrid, invlink.(fsamples); color, alpha=0.2, label="")
    return plot!(Xgrid, invlink.(mean(fx)); color, lw=2, label="posterior fit")
end

# Initialise the hyperparameters

theta0 = [0.0, 3.0]

lf = build_latent_gp(theta0)

lf.f.kernel

# Plot samples from approximate posterior

f_post = posterior(LaplaceApproximation(), lf(X), Y)

p1 = plot_data()
plot_samples!(Xgrid, f_post)

# ## Optimise the parameters

objective = build_laplace_objective(build_latent_gp, X, Y; newton_warmstart=true)

training_results = Optim.optimize(
    objective, θ -> only(Zygote.gradient(objective, θ)), theta0, LBFGS(); inplace=false
)

lf2 = build_latent_gp(training_results.minimizer)

lf2.f.kernel

# Plot samples from approximate posterior for optimised hyperparameters

f_post2 = posterior(LaplaceApproximation(; f_init=objective.f), lf2(X), Y)

p2 = plot_data()
plot_samples!(Xgrid, f_post2)
