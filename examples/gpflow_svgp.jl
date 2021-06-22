# An attempted recreation of https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

using AbstractGPs
using SparseGPs
using Distributions
using LinearAlgebra
using Optim
using IterTools

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)

# %%
function g(x)
    return sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)
end

N = 10000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

scatter(x, y; xlabel="x", ylabel="y", legend=false)

# %%
M = 50 # number of inducing points

# TODO: incorporate better inducing point selection from
# https://github.com/JuliaGaussianProcesses/InducingPoints.jl?
z = x[1:M]

# %%
# A simple Flux model
using Flux

struct SVGPLayer
    k # kernel parameters
    m # variational mean
    A # variational covariance
    z # inducing points
end

@Flux.functor SVGPLayer

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

function (m::SVGPLayer)(x)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    q = MvNormal(m.m, m.A'm.A)
    fx = f(x, 0.3)
    fu = f(m.z, 0.3)
    return fx, fu, q
end

function posterior(m::SVGPLayer)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    fu = f(m.z, 0.3)
    q = MvNormal(m.m, m.A'm.A)
    return SparseGPs.approx_posterior(SVGP(), fu, q)
end

function flux_loss(x, y; n_data=1, n_batch=1)
    fx, fu, q = model(x)
    return -SparseGPs.elbo(fx, y, fu, q; n_data, n_batch)
end

# Initialise the parameters
k = [0.3, 10]
m = zeros(M)
A = Matrix{Float64}(I, M, M)

model = SVGPLayer(k, m, A, z)

b = 100 # minibatch size
opt = ADAM(0.01)
# parameters = Flux.params(k, s, m, A)
parameters = Flux.params(model)
data_loader = Flux.Data.DataLoader((x, y), batchsize=b)

# %%
println(flux_loss(x, y))

Flux.train!(
    (x, y) -> flux_loss(x, y; n_data=N, n_batch=b),
    parameters,
    ncycle(data_loader, 100), # Train for 100 epochs
    opt
)

println(flux_loss(x, y))

# %%
post = posterior(model)

scatter(
    x,
    y;
    markershape=:xcross,
    markeralpha=0.1,
    xlim=(-1, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (VI with sparse grid)",
    label="Train Data",
)
plot!(-1:0.001:1, post; label="Posterior")
vline!(z; label="Pseudo-points")


# %% Find the exact posterior over u (e.g.
# https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/ equations
# (11) & (12)) As a sanity check.

function exact_q(fu, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fu, fx)
    Kuu = Symmetric(cov(fu))
    Σ = (Symmetric(cov(fu) + (1/σ²) * Kuf * Kuf'))
    m = ((1/σ²)*Kuu* (Σ\Kuf)) * y
    A = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, A)
end

kernel = make_kernel([0.2, 11])
f = GP(kernel)
fx = f(x, 0.1)
fu = f(z, 0.1)
q_ex = exact_q(fu, fx, y)

scatter(x, y)
scatter!(z, q_ex.μ)

# These two should be the same - and they are, the plot below shows almost identical predictions
ap_ex = SparseGPs.approx_posterior(SVGP(), fu, q_ex) # Hensman 2013 (exact) posterior
ap_tits = AbstractGPs.approx_posterior(VFE(), fx, y, fu) # Titsias posterior

# Should these be the same? (they currently aren't)
SparseGPs.elbo(fx, y, fu, q_ex)
AbstractGPs.elbo(fx, y, fu)

# %%
scatter(
    x,
    y;
    xlim=(0, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (VI with sparse grid)",
    label="Train Data",
)
plot!(-1:0.001:1, ap_ex; label="SVGP posterior")
plot!(-1:0.001:1, ap_tits; label="Titsias posterior")
vline!(z; label="Pseudo-points")

