# TODO: Remove this!
# COV_EXCL_START

struct SVGP{Tfz<:FiniteGP,Tq<:AbstractMvNormal}
    fz::Tfz
    q::Tq
end

raw"""
    posterior(svgp::SVGP)

Compute the approximate posterior [1] over the process `f =
svgp.fz.f`, given inducing inputs `z = svgp.fz.x` and a variational
distribution over inducing points `svgp.q` (which represents ``q(u)``
where `u = f(z)`). The approximate posterior at test points ``x^*``
where ``f^* = f(x^*)`` is then given by:

```math
q(f^*) = \int p(f | u) q(u) du
```
which can be found in closed form.

[1] - Hensman, James, Alexander Matthews, and Zoubin Ghahramani. "Scalable
variational Gaussian process classification." Artificial Intelligence and
Statistics. PMLR, 2015.
"""
function AbstractGPs.posterior(svgp::SVGP)
    q, fz = svgp.q, svgp.fz
    m, S = mean(q), _chol_cov(q)
    Kuu = _chol_cov(fz)
    B = Kuu.L \ S.L
    α = Kuu \ (m - mean(fz))
    data = (S=S, m=m, Kuu=Kuu, B=B, α=α)
    return ApproxPosteriorGP(svgp, fz.f, data)
end

function AbstractGPs.posterior(svgp::SVGP, fx::FiniteGP, ::AbstractVector)
    @assert svgp.fz.f === fx.f
    return posterior(svgp)
end

function Statistics.mean(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector)
    return mean(f.prior, x) + cov(f.prior, x, inducing_points(f)) * f.data.α
end

function Statistics.cov(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
end

function Statistics.var(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    return var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
end

function Statistics.cov(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector, y::AbstractVector)
    B = f.data.B
    Cxu = cov(f.prior, x, inducing_points(f))
    Cuy = cov(f.prior, inducing_points(f), y)
    D = f.data.Kuu.L \ Cuy
    E = Cxu / f.data.Kuu.L'
    return cov(f.prior, x, y) - (E * D) + (E * B * B' * D)
end

function StatsBase.mean_and_cov(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ = cov(f.prior, x) - At_A(D) + At_A(f.data.B' * D)
    return μ, Σ
end

function StatsBase.mean_and_var(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector)
    Cux = cov(f.prior, inducing_points(f), x)
    D = f.data.Kuu.L \ Cux
    μ = Cux' * f.data.α
    Σ_diag = var(f.prior, x) - diag_At_A(D) + diag_At_A(f.data.B' * D)
    return μ, Σ_diag
end

inducing_points(f::ApproxPosteriorGP{<:SVGP}) = f.approx.fz.x

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
function map(p::PosteriorFunctionSamples, x::AbstractVector)
    return map(e -> prior(p, e), x) + map(e -> update(p, e), x)
end

# x here is an *individual* input
prior(p::PosteriorFunctionSamples, x) = p.data.w'p.ϕ(x)
update(p::PosteriorFunctionSamples, x) = vec(p.data.v'p.data.cov_z(x))

function resample(p::PosteriorFunctionSamples, num_samples=p.num_samples)
    return PosteriorFunctionSamples(num_samples, p.ϕ, p.sample_w, p.v_from_w, p.cov_z)
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
    # ϕ: R^(N×D) -> R^(N×L)
    # size(w): (L, num_samples)
    ϕ, p_w = weight_space_approx(rng, f.prior.kernel, input_dims, feature_dims)

    sample_w(num_samples) = rand(rng, p_w, num_samples)

    z = f.approx.fz.x
    ϕz = ϕ.(z)
    function v_from_w(w)
        u = rand(rng, f.approx.q, num_samples)
        wtϕ = Base.map(p -> w'p, ϕz)
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
