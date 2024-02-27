module FiniteBasisModule

using KernelFunctions,LinearAlgebra, AbstractGPs, Random
import AbstractGPs: AbstractGP, FiniteGP
import Statistics
import ChainRulesCore

struct FiniteBasis <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(::FiniteBasis, d::Real) = d
KernelFunctions.metric(::FiniteBasis) = KernelFunctions.DotProduct()

struct DegeneratePosterior{P,T,C} <: AbstractGP
	prior::P
	w_mean::T
	w_prec::C
end

weight_form(A::KernelFunctions.ColVecs) = A.X'
weight_form(A::KernelFunctions.RowVecs) = A.X

function AbstractGPs.posterior(fx::FiniteGP{GP{M, B}}, y::AbstractVector{<:Real}) where {M, B <: FiniteBasis}
	kern = fx.f.kernel
	δ = y - mean(fx)
	X = weight_form(fx.x)
	X_prec = X' * inv(fx.Σy)
	Λμ = X_prec * y
	prec = cholesky(I + Symmetric(X_prec * X))
	w = prec \ Λμ
	DegeneratePosterior(fx.f, w, prec)
end

function Statistics.mean(f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(x)
	X * w
end

function Statistics.cov(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(x)
	AbstractGPs.Xt_invA_X(f.w_prec, X')
end

function Statistics.cov(f::DegeneratePosterior, x::AbstractVector, y::AbstractVector)
	X = weight_form(x)
	Y = weight_form(y)
	AbstractGPs.Xt_invA_Y(X', f.w_prec, Y')
end

function Statistics.var(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(x)
	AbstractGPs.diag_Xt_invA_X(f.w_prec, X')
end

function Statistics.rand(rng::AbstractRNG, f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(x)
	X * (f.w_prec.U \ randn(rng, length(x)))
end

struct RandomFourierFeature
	ws::Vector{Float64}
end

RandomFourierFeature(kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(k))
RandomFourierFeature(rng::AbstractRNG, kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(rng, k))

FFApprox(kern::Kernel, k::Int) = FiniteBasis() ∘ FunctionTransform(RandomFourierFeature(kern, k))
FFApprox(rng::AbstractRNG, kern::Kernel, k::Int) = FiniteBasis() ∘ FunctionTransform(RandomFourierFeature(rng, kern, k))


function (f::RandomFourierFeature)(x)
	Float64[cos.(f.ws .* x); sin.(f.ws .* x)] .* sqrt(2/length(f.ws))
end

end