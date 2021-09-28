# # Binary Classification with Laplace approximation
#
# This example demonstrates how to carry out non-conjugate Gaussian process
# inference using the Laplace approximation.
#
# For a basic introduction to the functionality of this library, please refer
# to the [User Guide](@ref).
#
# ## Setup

using ApproximateGPs
using LinearAlgebra
using Distributions
using LogExpFunctions: logistic, softplus, invsoftplus
using Zygote
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1);

# ## Generate training data
#
# We create a binary-labelled toy dataset:

Xgrid = -4:0.1:29  # for visualization
X = range(0, 23.5; length=48)  # training inputs
f(x) = 3 * sin(10 + 0.6x) + sin(0.1x) - 1  # latent function
fs = f.(X)  # latent function values at training inputs

lik = BernoulliLikelihood()  # has logistic invlink by default
## could use other invlink, e.g. normcdf(f) = cdf(Normal(), f)

invlink = lik.invlink
ps = invlink.(fs)  # probabilities at the training inputs
Y = [rand(Bernoulli(p)) for p in ps]  # observations at the training inputs
## could do this in one call as `Y = rand(lik(fs))`

function plot_data()
    plot(; xlims=extrema(Xgrid), xticks=0:6:24)
    plot!(Xgrid, invlink ∘ f; label="true probabilities")
    return scatter!(X, Y; label="observations", color=3)
end

plot_data()

# ## Create a latent GP
# 
# Here we write a function that creates our latent GP prior, given the
# hyperparameter vector `theta`. Compared to a "vanilla" GP, the `LatentGP`
# requires a function or functor that maps from the latent GP `f` to the
# distribution of observations `y`. This functor is commonly called "the
# likelihood".

function build_latent_gp(theta)
    ## `theta` is unconstrained, but kernel variance and lengthscale must be positive:
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])

    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)

    dist_y_given_f = BernoulliLikelihood()  # has logistic invlink by default
    ## We could also be explicit and define it as a function:
    ## dist_y_given_f(f) = Bernoulli(invlink(f))

    jitter = 1e-8  # required for numeric stability [TODO: where to explain this better?]
    return LatentGP(GP(kernel), dist_y_given_f, jitter)
end;

# We define a latent GP at our initial hyperparameter values, here with
# variance 1.0 and lengthscale 5.0:

theta0 = [invsoftplus(1.0), invsoftplus(5.0)]

lf = build_latent_gp(theta0)

lf.f.kernel

# We can now compute the Laplace approximation ``q(f)`` to the true posterior
# ``p(f | y)``:

f_post = posterior(LaplaceApproximation(), lf(X), Y)

# This finds the mode of the posterior (for the given values of the
# hyperparameters) using iterated Newton's method (i.e. solving an optimisation
# problem) and then constructs a Gaussian approximation to the posterior by
# matching the curvature at the mode.

# Let's plot samples from this approximate posterior:

function plot_samples!(Xgrid, fpost; samples=100, color=2)
    fx = fpost(Xgrid, 1e-8)
    fsamples = rand(fx, samples)
    plot!(Xgrid, invlink.(fsamples); color, alpha=0.2, label="")
    return plot!(Xgrid, invlink.(mean(fx)); color, lw=2, label="posterior fit")
end

p1 = plot_data()
plot_samples!(Xgrid, f_post)

# We can improve this fit by optimising the hyperparameters. For exact Gaussian
# process regression, the maximization objective is the marginal likelihood.
# Here, we can only optimise an _approximation_ to the marginal likelihood.

# ## Optimise the hyperparameters
#
# ApproximateGPs provides a convenience function `build_laplace_objective` that
# constructs an objective function for optimising the hyperparameters, based on
# the Laplace approximation to the log marginal likelihood.

objective = build_laplace_objective(build_latent_gp, X, Y);

# We pass this objective to Optim.jl's LBFGS optimiser:

training_results = Optim.optimize(
    objective, θ -> only(Zygote.gradient(objective, θ)), theta0, LBFGS(); inplace=false
)

# Now that we have (hopefully) better hyperparameter values, we need to construct a LatentGP prior with these values:

lf2 = build_latent_gp(training_results.minimizer)

lf2.f.kernel

# Finally, we need to construct again the (approximate) posterior given the
# observations for the latent GP with optimised hyperparameters:

f_post2 = posterior(LaplaceApproximation(; f_init=objective.f), lf2(X), Y)

# By passing `f_init=objective.f` we let the Laplace approximation "warm-start"
# at the last point of the inner-loop Newton optimisation; `objective.f` is a
# field on the `objective` closure.

# Let's plot samples from the approximate posterior for the optimised hyperparameters:

p2 = plot_data()
plot_samples!(Xgrid, f_post2)
