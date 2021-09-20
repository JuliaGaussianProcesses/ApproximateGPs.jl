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

function build_latent_gp(theta)
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return LatentGP(GP(kernel), dist_y_given_f, 1e-8)
end

function plot_samples!(Xgrid, fpost; samples=100, color=2)
    fsamples = rand(fpost(Xgrid, 1e-8), samples)
    plot!(Xgrid, invlink.(fsamples); color, alpha=0.3, label="")
end

using Zygote

theta0 = [0.0, 1.0]

optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)
optim_options = Optim.Options(; iterations=100)

lf = build_latent_gp(theta0)
lfX = lf(X)

newt_res = laplace_steps(lfX.lik, lfX.fx, Y)

f_post, opt_res = ApproximateGPs.optimize_elbo(build_latent_gp, theta0, X, Y, NelderMead(), optim_options)

theta1 = opt_res.minimizer

function full_objective(theta)
    Zygote.ignore() do
        # Zygote does not like the try/catch within @info
        @info "Hyperparameters: $theta"
    end
    lf = build_latent_gp(theta)
    f = zeros(length(X))
    lml = ApproximateGPs.laplace_lml!(f, lf(X), Y)
    return -lml
end

function nonewton_objective(theta)
    _lf = build_latent_gp(theta)
    return -ApproximateGPs.laplace_lml_nonewton(newt_res[end].f, _lf(X), Y)
end

using FiniteDifferences

FiniteDifferences.grad(central_fdm(5, 1), full_objective, theta0)


function comp_lml(theta)
    _lf = build_latent_gp(theta)
    K = kernelmatrix(_lf.f.kernel, X)
    dist_y_given_f = _lf.lik
    return ApproximateGPs.laplace_lml(K, dist_y_given_f, Y; maxiter=100)
end
