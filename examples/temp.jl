using ApproximateGPs
using AbstractGPs
using Distributions
using LinearAlgebra

k = 3 * (SqExponentialKernel() ∘ ScaleTransform(10))
gp = GP(k)
x = rand(9)
fx = gp(x, 0.01)
y = rand(fx)
C = rand(length(x), length(x))
ap = posterior(SVGP(fx, MvNormal(C'C)))

n_samples = 100

x_plot = sort(rand(100))

post_samples = ApproximateGPs.pathwise_sample(ap, x_plot, rand; num_samples=n_samples)

using Plots
plot(x_plot, post_samples; label="", color=:red, linealpha=0.2)
plot!(x_plot, ap; color=:green, label="True posterior")
