module FiniteBasisModule

using KernelFunctions,LinearAlgebra, AbstractGPs, ArraysOfArrays, Random
import AbstractGPs: AbstractGP, FiniteGP
import Statistics

struct FiniteBasis{T} <: Kernel
	ϕ::T
end

(k::FiniteBasis)(x, y) = dot(k.ϕ(x), k.ϕ(y))

struct DegeneratePosterior{P,T,C} <: AbstractGP
	prior::P
	w_mean::T
	w_prec::C
end

weight_form(ϕ, x) = flatview(ArrayOfSimilarArrays(ϕ.(x)))'

function AbstractGPs.posterior(fx::FiniteGP{GP{M, B}}, y::AbstractVector{<:Real}) where {M, B <: FiniteBasis}
	kern = fx.f.kernel
	δ = y - mean(fx)
	X = weight_form(kern.ϕ, fx.x)
	X_prec = X' * inv(fx.Σy)
	Λμ = X_prec * y
	prec = cholesky(I + Symmetric(X_prec * X))
	w = prec \ Λμ
	DegeneratePosterior(fx.f, w, prec)
end

function Statistics.mean(f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(f.prior.kernel.ϕ, x)
	X * w
end

function Statistics.cov(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(f.prior.kernel.ϕ, x)
	AbstractGPs.Xt_invA_X(f.w_prec, X')
end

function Statistics.cov(f::DegeneratePosterior, x::AbstractVector, y::AbstractVector)
	X = weight_form(f.prior.kernel.ϕ, x)
	Y = weight_form(f.prior.kernel.ϕ, y)
	AbstractGPs.Xt_invA_Y(X', f.w_prec, Y')
end

function Statistics.var(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(f.prior.kernel.ϕ, x)
	AbstractGPs.diag_Xt_invA_X(f.w_prec, X')
end

function Statistics.rand(rng::AbstractRNG, f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(f.prior.kernel.ϕ, x)
	X * (f.w_prec.U \ randn(rng, length(x)))
end

struct RandomFourierFeature
	ws::Vector{Float64}
end

RandomFourierFeature(kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(k))
RandomFourierFeature(rng::AbstractRNG, kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(rng, k))


function (f::RandomFourierFeature)(x)
	Float64[cos.(f.ws .* x); sin.(f.ws .* x)] .* sqrt(2/length(f.ws))
end

end