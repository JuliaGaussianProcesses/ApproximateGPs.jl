module NearestNeighborsModule

using KernelFunctions, LinearAlgebra, SparseArrays, AbstractGPs

function make_B(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
	n = length(pts)
	js = Int[]
	is = Int[]
	vals = T[]
	for i in 1:n
		if i == 1
			ns = T[]
		else
			ns = pts[max(1, i-k):i-1]
		end
		row = kernelmatrix(kern,ns) \ kern.(ns, pts[i])
		start_ix = max(i-k, 1)
		col_ixs = start_ix:(start_ix + length(row) - 1)
		append!(js, col_ixs)
		append!(is, fill(i, length(col_ixs)))
		append!(vals, row)
	end
	sparse(is, js, vals, n, n)
end

function make_F(pts::AbstractVector{T}, k::Int, kern::Kernel) where {T}
	n = length(pts)
	vals = T[]
	for i in 1:n
		prior = kern(pts[i], pts[i])
		if i == 1
			push!(vals, prior)
		else
			ns = pts[max(1, i-k):i-1]
			ki = kern.(ns, pts[i])
			push!(vals, prior - dot(ki, kernelmatrix(kern, ns) \ ki))
		end
	end
	Diagonal(vals)
end

struct NearestNeighbors
	k::Int
end

struct InvRoot{A}
	U::A
end

AbstractGPs.diag_Xt_invA_X(A::InvRoot, X::AbstractVecOrMat) = AbstractGPs.diag_At_A(A.U' * X)

AbstractGPs.Xt_invA_X(A::InvRoot, X::AbstractVecOrMat) = AbstractGPs.At_A(A.U' * X)

function AbstractGPs.posterior(nn::NearestNeighbors, fx::AbstractGPs.FiniteGP, y::AbstractVector)
	kern = fx.f.kernel
	F = make_F(fx.x, nn.k, kern)
	B = make_B(fx.x, nn.k, kern)
	U = UpperTriangular((I - B)' * inv(sqrt(F)))
	δ = y - mean(fx)
	α = U * (U' * δ)
	C = InvRoot(U)
	AbstractGPs.PosteriorGP(fx.f, (α=α, C=C, x=fx.x, δ=δ))
end

end