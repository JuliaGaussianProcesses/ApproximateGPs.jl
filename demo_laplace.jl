using Random
using LogExpFunctions
using Distributions
using Plots
using Optim
using AbstractGPs
using AbstractGPs: LatentFiniteGP
using ApproximateGPs

Random.seed!(1)
X = range(0, 23.5; length=48)
fs = @. 3*sin(10 + 0.6X) + sin(0.1X) - 1
# invlink = normcdf
invlink = logistic
ps = invlink.(fs)
Y = [rand(Bernoulli(p)) for p in ps]

function plot_data()
    plot()
    plot!(X, ps)
    scatter!(X, Y)
end

dist_y_given_f(f) = Bernoulli(invlink(f))

theta0 = [0.0, 1.0]

optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)
optim_options = Optim.Options(; iterations=100)

function build_latent_gp(theta)
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return LatentGP(GP(kernel), dist_y_given_f, 1e-8)
end
