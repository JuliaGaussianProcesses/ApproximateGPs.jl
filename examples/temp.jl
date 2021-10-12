# Not registered yet, so `add https://github.com/rossviljoen/RandomFourierFeatures.jl`
using RandomFourierFeatures
using ApproximateGPs
using AbstractGPs
using Distributions
using LinearAlgebra


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


k = 3 * (SqExponentialKernel() ∘ ScaleTransform(10))
gp = GP(k)

x = rand(20)
fx = gp(x, 0.01)
y = rand(fx)

z = x[1:8]
fz = gp(z)

q = exact_variational_posterior(fz, fx, y)
ap = posterior(SVGP(fz, q))

n_samples = 100

x_plot = sort(rand(500))

sample_fn = RandomFourierFeatures.create_prior_sample_function(1000)

function_samples = ApproximateGPs.pathwise_sample(ap, sample_fn; num_samples=n_samples)

y_samples = function_samples(x_plot)  # size(y_samples): (length(x_plot), n_samples)

using Plots
<<<<<<< HEAD
plot(x_plot, y_samples, label="", color=:red, linealpha=0.2)
plot!(x_plot, ap, color=:green, label="True posterior")
scatter!(x, y, label="data")
vline!(z, label="inducing points")
=======
plot(x_plot, post_samples; label="", color=:red, linealpha=0.2)
plot!(x_plot, ap; color=:green, label="True posterior")
>>>>>>> 97ecfb8bde36ecbb760bcd82286633a7d5ad262e
