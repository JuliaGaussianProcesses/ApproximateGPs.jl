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

struct PosteriorFunctionSamples
    num_samples
    prior  # prior(x) returns an approx prior sample
    update  # update(x) computes the required pathwise update
end

(f::PosteriorFunctionSamples)(x) = f([x])
(f::PosteriorFunctionSamples)(x::AbstractVector) = f.prior(x) + f.update(x)

# TODO: docstrings
function pathwise_sample(
    rng::AbstractRNG,
    f::ApproxPosteriorGP{<:SVGP},
    prior_sample_function; # TODO: better name?
    num_samples=1::Int
)
    svgp = f.approx
    z = svgp.fz.x
    input_dims = length(z[1])  # TODO: what's the best way of getting this?

    # Each function sample returns a fixed weight sample along with a L-dimensional feature mapping ϕ
    # ϕ: R^(N×D) -> R^(N×L)
    # size(w): (L, num_samples)
    ϕ, w = prior_sample_function(rng, f.prior, input_dims, num_samples)

    prior(x) = ϕ(x) * w

    u = rand(rng, svgp.q, num_samples)
    v = f.data.Kuu \ (u - ϕ(z) * w)
    function update(x)
        Kxu = cov(f.prior, x, z)
        return Kxu * v
    end

    return PosteriorFunctionSamples(num_samples, prior, update)
end

function pathwise_sample(
    f::ApproxPosteriorGP{<:SVGP}, prior_sample_function; num_samples=1::Int
)
    return pathwise_sample(Random.GLOBAL_RNG, f, prior_sample_function; num_samples)
end
