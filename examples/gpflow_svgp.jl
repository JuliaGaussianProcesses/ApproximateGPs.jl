# An attempted recreation of https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html

using AbstractGPs
using SparseGPs
using Distributions
using LinearAlgebra
using StatsFuns
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)

# %%
function g(x)
    return sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)
end

N = 1000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

scatter(x, y; xlabel="x", ylabel="y", legend=false)

# %%
M = 30 # number of inducing points

function pack_params(θ, m, A)
    return vcat(θ, m, vec(A))
end

function unpack_params(params, m; include_z=false)
    if include_z
        k = params[1:2]
        z = params[3:m+2]
        μ = params[m+3:2m+2]
        s = params[2m+3:end]
        Σ = reshape(s, (M, M))
        return k, z, μ, Σ
    else
        k = params[1:2]
        μ = params[3:m+2]
        s = params[m+3:end]
        Σ = reshape(s, (M, M))
        return k, μ, Σ
    end
end

x0 = pack_params(rand(2), zeros(M), vec(Matrix{Float64}(I, M, M)))
z = x[1:M]

# %%
function objective_function(x, y)
    function neg_elbo(params)
        # k, z, qμ, qΣ_L = split_params(params, M)
        k, m, A = unpack_params(params, M)
        kernel =
            (softplus(k[1])) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
        f = GP(kernel)
        fx = f(x, 0.1)
        q = MvNormal(m, A'A)
        return -SparseGPs.elbo(fx, y, f(z), q)
    end
    return neg_elbo
end

# Currently fails at the cholesky factorisation of cov(f(z))
opt = optimize(objective_function(x, y), x0, LBFGS())

# %%
opt_k, opt_μ, opt_Σ_L = unpack_params(opt.minimizer, M)
opt_kernel =
    softplus(opt_k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(opt_k[2]) + 0.01))
opt_f = GP(opt_kernel)
opt_q = MvNormal(opt_μ, opt_Σ_L * opt_Σ_L')
ap = SparseGPs.approx_posterior(SVGP(), opt_f(z), opt_q)
logpdf(ap(x), y)

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
# scatter!(x, y; label="Test Data")
plot!(-1:0.001:1, ap; label=false)
vline!(z; label="Pseudo-points")


# %% Find the exact posterior over u (e.g.
# https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/ equations
# (11) & (12)) As a sanity check -- this seems to work.

function exact_q(fu, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fu, fx)
    Kuu = Symmetric(cov(fu))
    Σ = (Symmetric(cov(fu) + (1/σ²) * Kuf * Kuf'))
    m = ((1/σ²)*Kuu* (Σ\Kuf)) * y
    A = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, A)
end

kernel = 0.3 * (SqExponentialKernel() ∘ ScaleTransform(10))
f = GP(kernel)
fx = f(x)
fu = f(z)
q_ex = exact_q(fu, fx, y)

scatter(x, y)
scatter!(z, q_ex.μ)

ap_ex = SparseGPs.approx_posterior(SVGP(), fu, q_ex)

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
plot!(-1:0.001:1, ap_ex; label=false)
vline!(z; label="Pseudo-points")
