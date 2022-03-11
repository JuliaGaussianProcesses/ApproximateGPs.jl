module PathwiseSamplingModule

export pathwise_sample

using Random

using ..ApproximateGPs: _chol_cov
using ..SparseVariationalApproximationModule:
    SparseVariationalApproximation, Centered, NonCentered, _get_q_u

using AbstractGPs:
    ApproxPosteriorGP, VFE, inducing_points, Xt_invA_X, Xt_A_X, inducing_points
using PDMats: chol_lower
using Distributions

struct PosteriorSample{Tapprox<:ApproxPosteriorGP,Tprior,Tv}
    approx_post::Tapprox  # The approximate posterior GP from which this sample is taken
    prior_sample::Tprior  # A function sampled from the prior of `approx_post`
    v::Tv  # The term needed to compute the pathwise update to the prior sample
end
function (s::PosteriorSample)(x::AbstractVector)
    return s.prior_sample(x) + cov(s.approx_post, x, inducing_points(s.approx_post)) * s.v
end

@doc raw"""
    pathwise_sample(rng::Random.AbstractRNG, f::ApproxPosteriorGP, weight_space_approx[, num_samples::Integer])

Efficiently samples a function from a sparse approximate posterior GP `f`.
Returns a function which can be evaluated at any input locations `X`.
`weight_space_approx` must be a function which takes a prior `AbstractGP` as an
argument and returns a `BayesianLinearRegressors.BasisFunctionRegressor`,
representing a weight space approximation to the prior of `f`. An example of
such a function can be constructed with
`RandomFourierFeatures.build_rff_weight_space_approx`.

If `num_samples` is supplied as an argument, returns a Vector of function
samples.

Details of the method can be found in [1].

[1] - Wilson, James, et al. "Efficiently sampling functions from Gaussian
process posteriors." International Conference on Machine Learning. PMLR, 2020.
"""
function pathwise_sample(rng::Random.AbstractRNG, f::ApproxPosteriorGP, weight_space_approx)
    prior_approx = weight_space_approx(f.prior)
    prior_sample = rand(rng, prior_approx)

    z = inducing_points(f)
    q_u = _get_q_u(f)

    u = rand(rng, q_u)
    v = cov(f, z) \ (u - prior_sample(z))

    return PosteriorSample(f, prior_sample, v)
end
pathwise_sample(f::ApproxPosteriorGP, wsa) = pathwise_sample(Random.GLOBAL_RNG, f, wsa)

function pathwise_sample(
    rng::Random.AbstractRNG, f::ApproxPosteriorGP, weight_space_approx, num_samples::Integer
)
    prior_approx = weight_space_approx(f.prior)
    prior_samples = rand(rng, prior_approx, num_samples)

    z = inducing_points(f)
    q_u = _get_q_u(f)

    us = rand(rng, q_u, num_samples)

    vs = cov(f, z) \ (us - reduce(hcat, map((s) -> s(z), prior_samples)))

    posterior_samples = [
        PosteriorSample(f, s, v) for (s, v) in zip(prior_samples, eachcol(vs))
    ]
    return posterior_samples
end
function pathwise_sample(f::ApproxPosteriorGP, wsa, num_samples::Integer)
    return pathwise_sample(Random.GLOBAL_RNG, f, wsa, num_samples)
end

end
