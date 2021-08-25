# # Classification: Sparse Variational Approximation for Non-Conjugate Likelihoods with Optim's L-BFGS
# 
#
# This example demonstrates how to carry out non-conjugate Gaussian process
# inference using the stochastic variational Gaussian process (SVGP) model. For
# a basic introduction to the functionality of this library, please refer to the
# [User Guide](@ref).
#
# ## Setup

using SparseGPs
using ParameterHandling
using Zygote
using Distributions
using LinearAlgebra
using Optim

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)
#md nothing #hide

# ## Generate some training data
#
# For our binary classification model, we will use the standard approach of a
# latent GP with a Bernoulli likelihood. This results in a generative model that
# we can use to produce some training data.
#
# First, we define the underlying latent GP
# ```math
# f \sim \mathcal{GP}(0, k(\cdot, \cdot'))
# ```
# and sample a function `f`.

k_true = [30.0, 0.5]
kernel_true = k_true[1] * (SqExponentialKernel() ∘ ScaleTransform(k_true[2]))

jitter = 1e-12  # for numeric stability
lgp = LatentGP(GP(kernel_true), BernoulliLikelihood(), jitter)
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

N = 30  # The number of training points
mask = sample(1:length(x_true), N; replace=false, ordered=true)  # Subsample some input locations
x, y = x_true[mask], y_true[mask]

scatter(x, y; label="Sampled outputs")
plot!(x_true, mean.(lgp.lik.(f_true)); seriescolor="red", label="True mean")

# ## Creating an SVGP
#
# Now that we have some data sampled from a generative model, we can try to recover the
# true generative function with an SVGP classification model.
#
# For this, we shall use a mixture of
# [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl) to
# deal with our constrained parameters and
# [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) to perform
# optimimisation.
#
# The required parameters for the SVGP are - the kernel hyperparameters `k`, the
# inducing inputs `z` and the mean and covariance of the variational
# distribution `q`; given by `m` and `A` respectively. ParameterHandling
# provides an elegant way to deal with the constraints on these parameters,
# since `k` must be positive and `A` must be positive definite. For more
# details, see the
# [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl)
# readme.

# First, we need to define a quick and dirty positive definite matrix type for
# ParameterHandling.jl - this code can safely be ignored.

struct PDMatrix{TA}
    A::TA
end

function pdmatrix(A::AbstractMatrix)
    return PDMatrix(A)
end

function ParameterHandling.value(P::PDMatrix)
    A = copy(P.A)
    return A'A
end

function ParameterHandling.flatten(::Type{T}, P::PDMatrix) where {T}
    v, unflatten_to_Array = flatten(T, P.A)
    function unflatten_PDmatrix(v_new::Vector{T})
        A = unflatten_to_Array(v_new)
        return PDMatrix(A)
    end
    return v, unflatten_PDmatrix
end
#md nothing #hide

# Initialise the parameters

M = 15  # number of inducing points
raw_initial_params = (
    k=(var=positive(rand()), precision=positive(rand())),
    z=bounded.(range(0.1, 5.9; length=M), 0.0, 6.0),  # constrain z to simplify optimisation
    m=zeros(M),
    A=pdmatrix(4 * Matrix{Float64}(I, M, M)),  # pdmatrix is defined in utils.jl
)
#md nothing #hide

# `flatten` takes the `NamedTuple` of parameters and returns a flat vector of
# `Float64` - along with a function `unflatten` to reconstruct the `NamedTuple`
# from a flat vector. `value` takes each parameter in the `NamedTuple` and
# applies the necessary transformation to return the constrained value which can
# then be used to construct the SVGP model. `unpack` therefore takes a flat,
# unconstrained `Vector{Float64}` and returns a `NamedTuple` of constrained
# parameter values.

flat_init_params, unflatten = ParameterHandling.flatten(raw_initial_params)
unpack = ParameterHandling.value ∘ unflatten
#md nothing #hide

# Now, we define a function to build the SVGP model from the constrained
# parameters as well as a loss function - in this case the negative ELBO.

lik = BernoulliLikelihood()
jitter = 1e-3  # added to aid numerical stability

function build_SVGP(params::NamedTuple)
    kernel = params.k.var * (SqExponentialKernel() ∘ ScaleTransform(params.k.precision))
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(params.m, params.A)
    fz = f(params.z).fx
    return SVGP(fz, q), f
end

function loss(params::NamedTuple)
    svgp, f = build_SVGP(params)
    fx = f(x)
    return -elbo(svgp, fx, y)
end
#md nothing #hide

# Optimise the parameters using LBFGS.

opt = optimize(
    loss ∘ unpack,
    θ -> only(Zygote.gradient(loss ∘ unpack, θ)),
    flat_init_params,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=4_000);
    inplace=false,
)

# Finally, build the optimised SVGP model, and sample some functions to see if
# they are close to the true function.

final_params = unpack(opt.minimizer)

svgp_opt, f_opt = build_SVGP(final_params)
post_opt = posterior(svgp_opt)
l_post_opt = LatentGP(post_opt, BernoulliLikelihood(), jitter)

post_f_samples = rand(l_post_opt.f(x_true, 1e-6), 20)
post_μ_samples = mean.(l_post_opt.lik.(post_f_samples))

plt = plot(x_true, post_μ_samples; seriescolor="red", linealpha=0.2, label="")
scatter!(plt, x, y; seriescolor="blue", label="Data points")
vline!(final_params.z; label="Pseudo-points")
plot!(
    x_true, mean.(lgp.lik.(f_true)); seriescolor="green", linewidth=3, label="True function"
)
