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

struct SVGPModel
    k # kernel parameters
    m # variational mean
    A # variational covariance
    z # inducing points
end

@Flux.functor SVGPModel (k, m, A,) # Don't train the inducing inputs

function make_kernel(k)
    return softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
end

# Create the 'model' from the parameters - i.e. return the FiniteGP at inputs x,
# the FiniteGP at inducing inputs z and the variational posterior over inducing
# points - q(u).
function (m::SVGPModel)(x)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    q = MvNormal(m.m, m.A'm.A)
    fx = f(x, 0.3)
    fu = f(m.z, 0.3)
    return fx, fu, q
end

# Create the posterior GP from the model parameters.
function posterior(m::SVGPModel)
    kernel = make_kernel(m.k)
    f = GP(kernel)
    fu = f(m.z, 0.3)
    q = MvNormal(m.m, m.A'm.A)
    return SparseGPs.approx_posterior(SVGP(), fu, q)
end

# Return the loss given data - in this case the negative ELBO.
function flux_loss(x, y; n_data=length(y))
    fx, fu, q = model(x)
    return -SparseGPs.elbo(fx, y, fu, q; n_data)
end


# %%
M = 50 # number of inducing points

# Select the first M inputs as inducing inputs
z = x[1:M]

# Initialise the parameters
k = [0.3, 10]
m = zeros(M)
A = Matrix{Float64}(I, M, M)

model = SVGPModel(k, m, A, z)

b = 100 # minibatch size
opt = ADAM(0.01)
parameters = Flux.params(model)
data_loader = Flux.Data.DataLoader((x, y), batchsize=b)

# %%
# Negative ELBO before training
println(flux_loss(x, y))

# %%
# Train the model
Flux.train!(
    (x, y) -> flux_loss(x, y; n_data=N),
    parameters,
    ncycle(data_loader, 300), # Train for 300 epochs
    opt
)

# %%
# Negative ELBO after training
println(flux_loss(x, y))

# %%
# Plot samples from the optmimised approximate posterior.
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

kernel = make_kernel([0.2, 11])
f = GP(kernel)
fx = f(x, 0.1)
fu = f(z, 0.1)
q_ex = exact_q(fu, fx, y)

scatter(x, y)
scatter!(z, q_ex.μ)

# These two should be the same - and they are, as the plot below shows
ap_ex = SparseGPs.approx_posterior(SVGP(), fu, q_ex) # Hensman (2013) exact posterior
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

