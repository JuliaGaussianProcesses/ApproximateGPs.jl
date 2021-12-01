# # Augmented Sparse Variational Approximation for Non-Conjugate Likelihoods,
# # a Classification Example
# 
#
# This example demonstrates how to use augmentation to improve convergence speed
# for variational Gaussian process with non-conjugate likelihoods.
# inference using the stochastic variational Gaussian process (SVGP) model. For
# a basic introduction to the functionality of this library, please refer to the
# [User Guide](@ref).
#
# ## Setup

using ApproximateGPs
using ParameterHandling
using Zygote
using Distributions
using LinearAlgebra
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

# We import AGP.jl for the Polya-Gamma distribution
using AugmentedGaussianProcesses.ComplementaryDistributions: PolyaGamma

using Random
Random.seed!(1234);

# ## Generate some training data
#
# We work in the same setting as the other [classification example](../b-classification).
# For our binary classification model, we will use the standard approach of a
# latent GP with a Bernoulli likelihood. This results in a generative model that
# we can use to produce some training data.
#
# First, we define the underlying latent GP
# ```math
# f \sim \mathcal{GP}(0, k(\cdot, \cdot'))
# ```
# and sample a function `f`.

k_true = [30.0, 1.5]
kernel_true = k_true[1] * (Matern32Kernel() ∘ ScaleTransform(k_true[2]))

jitter = 1e-6  # for numeric stability
lik = BernoulliLikelihood()
lgp = LatentGP(GP(kernel_true), lik, jitter)
x_true = 0:0.02:6
f_true, y_true = rand(lgp(x_true))

plot(x_true, f_true; seriescolor="red", label="")  # Plot the sampled function

# Then, the output of this sampled function is pushed through a logistic sigmoid
# `μ = σ(f)` to constrain the output to `[0, 1]`.

μ = mean.(lgp.lik.(f_true))
plot(x_true, μ; seriescolor="red", label="")

# Finally, the outputs `y` of the process are sampled from a Bernoulli
# distribution with mean `μ`. We're only interested in the outputs at a subset
# of inputs `x`, so we first pick some random input locations and then find the
# corresponding values for `y`.

N = 200  # The number of training points
mask = sample(1:length(x_true), N; replace=false, ordered=true)  # Subsample some input locations
x, y = x_true[mask], y_true[mask]

scatter(x, y; label="Sampled outputs")
plot!(x_true, mean.(lgp.lik.(f_true)); seriescolor="red", label="True mean")

# Let's first see how we can obtain the optimal variational parameters
# for our variational distribution q(u) = N(m, S).
# We will for now use the true kernel parameters,
# We define the inducing points locations and the associated `FiniteGP`
M = 15;
z = range(0.1, 5.9; length=M);
fz = GP(kernel_true)(z, jitter);

# We initialize the variational parameters q(u) = N(m, S) and q(ωᵢ) = p(ωᵢ) = PolyaGamma(1,0)
m = zeros(M);
S = Matrix{Float64}(I(M));
qω = [PolyaGamma(1, 0.0) for _ in 1:length(x)];

# Following the derivations from the paper [Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383),
# we build the following `cavi!` function which will update the parameters `m` and `S` as well as the variables `ω`

function cavi!(fz::AbstractGPs.FiniteGP, x, y, m, S, qω; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    κ = K \ cov(fz.f, fz.x, x)
    y_sign = sign.(y .- 0.5)
    for _ in 1:niter
        postu = posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
        postfs = marginals(postu(x))
        marginals_to_aug_posterior!(qω, postfs)
        S .= inv(Symmetric(inv(K) + project(Diagonal(mean.(qω)), κ)))
        m .= S * (project(y_sign / 2, κ) - K \ mean(fz))
    end
    return m, S, qω
end

function marginals_to_aug_posterior!(qω::AbstractVector, ps::AbstractVector{<:Normal})
    map!(qω, ps) do p
        PolyaGamma(1, sqrt(abs2(mean(p)) + var(p)))
    end
end

# where `project` which project the noise to the right dimension
project(x::AbstractVector, κ) = κ * x
project(X::AbstractMatrix, κ) = κ * X * κ'

# Let's now compare both results
post_svgp_init = posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
cavi!(fz, x, y, m, S, qω) # m, S and qω are going to be modified
post_svgp_opt = posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))

pf = scatter(x, y; legend=false, xlabel="x", ylabel="f")
plot!(x_true, f_true; seriescolor="red", label="True latent")
plot!(x_true, post_svgp_init(x_true); label="Initial Posterior")
plot!(x_true, post_svgp_opt(x_true); label="Final Posterior")
py = scatter(x, y; label="Training data", xlabel="x", ylabel="y")
plot!(x_true, mean.(lik.(f_true)); label="True mean")
plot!(x_true, mean.(lik.(mean(post_svgp_init(x_true)))); label="Initial Posterior")
plot!(x_true, mean.(lik.(mean(post_svgp_opt(x_true)))); label="Final Posterior")
plot(pf, py)
# ## Optimizing the hyperparameters

# What if like in the [classical optimization case](../b-classification)
# we want to optimize hyperparameters as well?
# Let's use the same approach!
raw_initial_params = (
    k=(var=positive(rand()), precision=positive(rand())),
    z=bounded.(range(0.1, 5.9; length=M), 0.0, 6.0),  # constrain z to simplify optimisation
);
flat_init_params, unflatten = ParameterHandling.flatten(raw_initial_params)
unpack = ParameterHandling.value ∘ unflatten;

# We now need to pass our variational parameters and modify the elbo
# to be able to infer correctly the results.
# This means we will have to implicitly pass the variational parameters
# `var_params` and update them after each update of the hyperparameters

# ### The augmented ELBO
function augmented_elbo(
    sva::SparseVariationalApproximation,
    lfx::AbstractGPs.LatentFiniteGP,
    y::AbstractVector,
    qω::AbstractVector;
    num_data=length(y),
)
    post = posterior(sva)
    q_f = marginals(post(lfx.fx.x))
    variational_exp = expected_aug_loglik(y, q_f, lfx.lik, qω)

    n_batch = length(y)
    scale = num_data / n_batch
    # We ignore the KL divergence regarding the augmented variables as it does
    # not depend on the kernel parameters
    return sum(variational_exp) * scale - Zygote.@ignore(kl_term(lik, qω)) -
           ApproximateGPs.kl_term(sva, post)
end

function kl_term(::BernoulliLikelihood{<:LogisticLink}, qω::AbstractVector{<:PolyaGamma})
    sum(qω) do q
        c = q.c
        -abs2(c) * mean(q) + 2log(cosh(c / 2))
    end
end

function expected_aug_loglik(y, q_f, ::BernoulliLikelihood, qω)
    return map(y, q_f, qω) do y, q, qω
        m = mean(q)
        sign(y - 0.5) * m / 2 - (abs2(m) + var(q)) * mean(qω) / 2 - log(2)
    end
end

# ### The complete inference process
function loss(params::NamedTuple)
    fz, lf = build_fz_lf(params)
    fx = lf(x)
    # We ignore this part as it involves inplace operations
    # and does not contribute to the final elbo
    Zygote.@ignore cavi!(fz, x, y, m, S, qω; niter=1)
    augsvgp = SparseVariationalApproximation(Centered(), fz, MvNormal(m, Symmetric(S)))
    return -augmented_elbo(augsvgp, fx, y, qω)
end

function build_fz_lf(params::NamedTuple)
    kernel = params.k.var * (Matern32Kernel() ∘ ScaleTransform(params.k.precision))
    lf = LatentGP(GP(kernel), lik, jitter)
    fz = GP(kernel)(params.z)
    return fz, lf
end

opt = optimize(
    loss ∘ unpack,
    θ -> only(Zygote.gradient(loss ∘ unpack, θ)),
    flat_init_params,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=100);
    inplace=false,
)

# And we can now compare with the optimal result we obtained earlier
θ_opt = unpack(opt.minimizer)
fz_opt, lf_opt = build_fz_lf(θ_opt)
post_svgp_hpopt = posterior(
    SparseVariationalApproximation(Centered(), fz_opt, MvNormal(m, S))
)

pf = scatter(x, y; legend=false, xlabel="x", ylabel="f")
plot!(x_true, f_true; seriescolor="red", label="True latent")
plot!(x_true, post_svgp_opt(x_true); label="Posterior")
plot!(x_true, post_svgp_hpopt(x_true); label="Posterior (hp opt)")
py = scatter(x, y; label="Training data", xlabel="x", ylabel="y")
plot!(x_true, mean.(lik.(f_true)); label="True mean")
plot!(x_true, mean.(lik.(mean(post_svgp_opt(x_true)))); label="Posterior")
plot!(x_true, mean.(lik.(mean(post_svgp_hpopt(x_true)))); label="Posterior (hp opt)")
plot(pf, py)