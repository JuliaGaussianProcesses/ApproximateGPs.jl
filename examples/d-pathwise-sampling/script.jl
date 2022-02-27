# ## Setup

using RandomFourierFeatures
using ApproximateGPs
using AbstractGPs
using Distributions
using LinearAlgebra
using Random

rng = MersenneTwister(1234)

# Find the exact Titsias posterior (avoid optimisation)
# TODO: this is already (and better) solved by `_get_q_u(f::ApproxPosteriorGP{<:VFE})`
# Replace this (and the version in `test_utils`?)
function optimal_variational_posterior(::Centered, fz, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fz, fx)
    Kuu = Symmetric(cov(fz))
    Σ = (Symmetric(cov(fz) + (1 / σ²) * Kuf * Kuf'))
    m = ((1 / σ²) * Kuu * (Σ \ Kuf)) * y
    S = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, S)
end

function optimal_variational_posterior(::NonCentered, fz, fx, y)
    q = optimal_variational_posterior(Centered(), fz, fx, y)
    Cuu = cholesky(Symmetric(cov(fz)))
    return MvNormal(Cuu.L \ (mean(q) - mean(fz)), Symmetric((Cuu.L \ cov(q)) / Cuu.U))
end

# Define a GP and generate some data

k = 3 * (SqExponentialKernel() ∘ ScaleTransform(10))
gp = GP(k)

input_dims = 1

x = ColVecs(rand(input_dims, 20))
fx = gp(x, 0.01)
y = rand(fx)

z = x[1:8]
fz = gp(z)

# Any of the following will work:

# q = optimal_variational_posterior(NonCentered(), fz, fx, y)
# ap = posterior(SparseVariationalApproximation(NonCentered(), fz, q))

# q = optimal_variational_posterior(Centered(), fz, fx, y)
# ap = posterior(SparseVariationalApproximation(NonCentered(), fz, q))

ap = posterior(VFE(fz), fx, y)

x_test = ColVecs(sort(rand(input_dims, 500); dims=2))

num_features = 1000
rff_wsa = build_rff_weight_space_approx(rng, input_dims, num_features)

function_samples = [ApproximateGPs.pathwise_sample(rng, ap, rff_wsa) for _ in 1:100]

y_samples = reduce(hcat, map((f) -> f(x_test), function_samples))  # size(y_samples): (length(x_plot), n_samples)

# Plot sampled functions against the exact posterior

using Plots

x_plot = x_test.X'

plot(x_plot, y_samples; label="", color=:red, linealpha=0.2)
plot!(vec(x_plot), ap; color=:blue, label="True posterior")
scatter!(x.X', y; label="data")
vline!(vec(z.X'); label="inducing points", color=:orange)
