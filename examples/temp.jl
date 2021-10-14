# TODO: Remove this!
# COV_EXCL_START

# A quick demonstration of the current pathwise sampling API
# Meant to just be run as a script, but can turn it into a notebook with:
# jupytext --to ipynb temp.jl

# %%
# Not registered yet, so `add https://github.com/rossviljoen/RandomFourierFeatures.jl`
using RandomFourierFeatures
using ApproximateGPs
using AbstractGPs
using Distributions
using LinearAlgebra
using Random

rng = MersenneTwister(1234)

# Find the exact Titsias posterior (avoid optimisation)
function exact_variational_posterior(fu, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fu, fx)
    Kuu = Symmetric(cov(fu))
    Σ = (Symmetric(cov(fu) + (1 / σ²) * Kuf * Kuf'))
    m = ((1 / σ²) * Kuu * (Σ \ Kuf)) * y
    S = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, S)
end

# %%
k = 3 * (SqExponentialKernel() ∘ ScaleTransform(10))
gp = GP(k)

input_dims = 1

x = rand(input_dims, 20)
fx = gp(x, 0.01)
y = rand(fx)

z = x[:, 1:8]
fz = gp(z)

# %%
q = exact_variational_posterior(fz, fx, y)
ap = posterior(SVGP(fz, q))

num_samples = 100

x_test = sort(rand(input_dims, 500); dims=2)

feature_dims = 1000
sample_fn = RandomFourierFeatures.gp_rff_approx

function_samples = ApproximateGPs.pathwise_sample(
    ap, sample_fn, input_dims, feature_dims; num_samples=num_samples
)

y_samples = function_samples.(ColVecs(x_test))  # size(y_samples): (length(x_plot), n_samples)

# %%
using Plots

x_plot = vec(x_test')

plot(x_plot, hcat(y_samples...)'; label="", color=:red, linealpha=0.2)
plot!(x_plot, ap; color=:green, label="True posterior")
scatter!(vec(x'), y; label="data")
vline!(vec(z'); label="inducing points")

# %%
# TODO: Remove this!
# COV_EXCL_STOP
