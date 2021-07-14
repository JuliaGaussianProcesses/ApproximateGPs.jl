# A recreation of https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

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
# The data generating function
function g(x)
    return sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)
end

N = 10000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

scatter(x, y; xlabel="x", ylabel="y", legend=false)


# %%
# A simple Flux model
using Flux

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

# %%
M = 50 # number of inducing points

# Select the first M inputs as inducing inputs
z = x[1:M]

# Initialise the kernel parameters
k = [0.3, 10]

model = SVGPModel(make_kernel, k, z; likelihood=GaussianLikelihood(0.1))

b = 100 # minibatch size
opt = ADAM(0.001)
parameters = Flux.params(model)
delete!(parameters, model.z)    # Don't train the inducing inputs
data_loader = Flux.Data.DataLoader((x, y), batchsize=b)

# %%
# Negative ELBO before training
println(loss(model, x, y))

# %%
# Train the model
Flux.train!(
    (x, y) -> loss(model, x, y, n_data=N),
    parameters,
    ncycle(data_loader, 300), # Train for 300 epochs
    opt
)

# %%
# Negative ELBO after training
println(loss(model, x, y))

# %%
# Plot samples from the optimised approximate posterior.
post = SparseGPs.posterior(model)

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


# %% There is a closed form optimal solution for the variational posterior q(u)
# (e.g. https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
# equations (11) & (12)). The SVGP posterior with this optimal q(u) should
# therefore be equivalent to the 'exact' sparse GP (Titsias) posterior.

function exact_q(fu, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fu, fx)
    Kuu = Symmetric(cov(fu))
    Σ = (Symmetric(cov(fu) + (1/σ²) * Kuf * Kuf'))
    m = ((1/σ²)*Kuu* (Σ\Kuf)) * y
    S = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, S)
end

kernel = kernel = make_kernel([0.3, 10])
f = GP(kernel)
fx = f(x, 0.1)
fu = f(z, 1e-6)

q_ex = exact_q(fu, fx, y)

scatter(x, y)
scatter!(z, q_ex.μ)

# These two should be the same - and they are, as the plot below shows
ap_ex = SparseGPs.approx_posterior(SVGP(), fu, q_ex) # Hensman (2013) exact posterior
ap_tits = AbstractGPs.approx_posterior(VFE(), fx, y, fu) # Titsias posterior

# These are also approximately equal
SparseGPs.elbo(fx, y, fu, q_ex)
AbstractGPs.elbo(fx, y, fu)

# %%
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
plot!(-1:0.001:1, ap_ex; label="SVGP posterior")
plot!(-1:0.001:1, ap_tits; label="Titsias posterior")
vline!(z; label="Pseudo-points")
