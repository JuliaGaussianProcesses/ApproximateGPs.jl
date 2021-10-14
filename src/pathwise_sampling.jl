# TODO: Remove this!
# COV_EXCL_START

## Pathwise posterior sampling
# TODO: docstrings

struct PosteriorFunctionSamples{Tϕ,Tdata}
    num_samples::Int
    ϕ::Tϕ
    data::Tdata
end

function PosteriorFunctionSamples(num_samples, ϕ, sample_w, v_from_w, cov_z)
    w = sample_w(num_samples)
    v = v_from_w(w)
    data = (w=w, v=v, sample_w=sample_w, v_from_w=v_from_w, cov_z=cov_z)
    return PosteriorFunctionSamples(num_samples, ϕ, data)
end

(p::PosteriorFunctionSamples)(x) = prior(p, x) + update(p, x)
function Base.map(p::PosteriorFunctionSamples, x::AbstractVector)
    return map(e -> prior(p, e), x) + map(e -> update(p, e), x)
end

# x here is an *individual* input
prior(p::PosteriorFunctionSamples, x) = p.data.w'p.ϕ(x)
update(p::PosteriorFunctionSamples, x) = vec(p.data.v'p.data.cov_z(x))

function resample(p::PosteriorFunctionSamples; num_samples=p.num_samples)
    return PosteriorFunctionSamples(
        num_samples, p.ϕ, p.data.sample_w, p.data.v_from_w, p.data.cov_z
    )
end

function pathwise_sample(
    rng::AbstractRNG,
    f::ApproxPosteriorGP{<:SVGP},
    weight_space_approx,
    input_dims,
    feature_dims;
    num_samples=1::Int,
)
    # weight_space_approx returns an L-dimensional feature mapping ϕ along with
    # a distribution over weights p_w such that ̃f(x) = wᵀϕ(x) approximates the
    # prior GP.
    # ϕ: R^D -> R^L
    # size(w): (L, num_samples)
    ϕ, p_w = weight_space_approx(rng, f.prior.kernel, input_dims, feature_dims)

    sample_w(num_samples) = rand(rng, p_w, num_samples)

    z = f.approx.fz.x
    ϕz = ϕ.(z)
    # `v` is a cached term needed for the pathwise `update` - if `w` is updated, `v` must be too
    function v_from_w(w)
        u = rand(rng, f.approx.q, num_samples)
        wtϕ = map(p -> w'p, ϕz)
        return f.data.Kuu \ (u - hcat(wtϕ...)')
    end

    cov_z(x) = cov(f.prior, z, [x])

    return PosteriorFunctionSamples(num_samples, ϕ, sample_w, v_from_w, cov_z)
end

function pathwise_sample(
    f::ApproxPosteriorGP{<:SVGP},
    prior_sample_function,
    input_dims,
    feature_dims;
    num_samples=1::Int,
)
    return pathwise_sample(
        Random.GLOBAL_RNG, f, prior_sample_function, input_dims, feature_dims; num_samples
    )
end

# TODO: Remove this!
# COV_EXCL_STOP
