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
using ParameterHandling
using Zygote
using Distributions
using LogExpFunctions
using LinearAlgebra
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1);

# ## Generate some training data

Xgrid = -5:0.1:29
X = range(0, 23.5; length=48)
fs = @. 3 * sin(10 + 0.6X) + sin(0.1X) - 1
# invlink = normcdf
invlink = logistic
ps = invlink.(fs)
Y = [rand(Bernoulli(p)) for p in ps]

function plot_data()
    plot()
    plot!(X, ps)
    return scatter!(X, Y)
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
    fsamples = rand(fpost(Xgrid, 1e-8), samples)
    return plot!(Xgrid, invlink.(fsamples); color, alpha=0.3, label="")
end

# Initialise the hyperparameters

theta0 = [0.0, 1.0]

lf = build_latent_gp(theta0)

lf.f.kernel

# Plot samples from approximate posterior

f_post = posterior(LaplaceApproximation(), lf(X), Y)

plot_data()
plot_samples!(Xgrid, f_post)

# ## Optimise the parameters

objective = build_laplace_objective(build_latent_gp, X, Y; newton_warmstart=true)

training_results = Optim.optimize(
    objective, θ -> only(Zygote.gradient(objective, θ)), theta0, LBFGS(); inplace=false
)

lf2 = build_latent_gp(training_results.minimizer)

lf2.f.kernel

# Plot samples from approximate posterior for optimised hyperparameters

f_post2 = posterior(LaplaceApproximation(), lf(X), Y)

plot_data()
plot_samples!(Xgrid, f_post2)
