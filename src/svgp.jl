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

raw"""
    pathwise_sample(
        [rng::AbstractRNG,]
        f::ApproxPosteriorGP{<:SVGP},
        x::AbstractVector,
        prior_sample_function,
        num_samples=1::Int
    )

Takes a 'pathwise sample' from the posterior GP `f` at `x` by using Matheron's
rule [2]. This works by taking `num_samples` (possibly approximate) samples from
`f.prior` and then updating these prior samples with samples `u` from
`f.approx.q` by Matheron's rule:

```math
f^* | u = f^* + K_{*, u} K_{u, u}^{-1} (u - f^z)
```

where ``f^* = f(x^*)`` and ``f^z = f(z)`` are the prior samples at the test
points and inducing points respectively. ``u \sim q(u)`` is a sample from the
variational posterior. ``K_{u, u}`` and ``K_{*, u}`` are the prior covariances
for the inducing points and for between the test and inducing points
respectively.

`prior_sample_function` can be any function which returns exact or approximate
samples from the prior. It must have the signature:

`prior_sample_function(rng::AbstractRNG, fx::FiniteGP, num_samples::Int)`

[2] - James T. Wilson and Viacheslav Borovitskiy and Alexander Terenin and Peter
Mostowsky and Marc Peter Deisenroth. "Efficiently Sampling Functions from
Gaussian Process Posteriors" ICML, 2020.
"""
function pathwise_sample(
    rng::AbstractRNG,
    f::ApproxPosteriorGP{<:SVGP},
    x::AbstractVector,
    prior_sample_function;
    num_samples=1::Int
)
    svgp = f.approx
    z = svgp.fz.x
    Kxu = cov(f.prior, x, z)
    Kuu = f.data.Kuu
    u = rand(rng, svgp.q, num_samples)

    # Jointly sample the prior at both the test points x^* and inducing inputs z
    prior_sample = prior_sample_function(rng, f.prior(vcat(x, z), 1e-8), num_samples)

    # Split the prior sample into f^* and f^z
    f_star = selectdim(prior_sample, 1, 1:size(x, 1))
    f_z = selectdim(prior_sample, 1, size(x, 1)+1:size(prior_sample, 1))

    # Apply Matheron's rule
    return f_star + Kxu * (Kuu \ (u - f_z))
end

function pathwise_sample(f::ApproxPosteriorGP{<:SVGP}, x::AbstractVector, prior_sample_function; num_samples=1::Int)
    pathwise_sample(Random.GLOBAL_RNG, f, x, prior_sample_function; num_samples)
end

inducing_points(f::ApproxPosteriorGP{<:SVGP}) = f.approx.fz.x

_chol_cov(q::AbstractMvNormal) = cholesky(Symmetric(cov(q)))
_chol_cov(q::MvNormal) = cholesky(q.Σ)
