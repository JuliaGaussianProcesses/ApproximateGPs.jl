module NearestNeighborsModule
using ..API

using KernelFunctions, LinearAlgebra, SparseArrays, AbstractGPs

@doc raw"""
Constructs the matrix $B$ for which $f = Bf + \epsilon$ were $f$
are the values of the GP and $\epsilon$ is a vector of zero mean
independent Gaussian noise. 
This matrix builds the conditional mean for function value $f_i$
in terms of the function values for previous $f_j$, where $j < i$.
See equation (9) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
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
	return sparse(is, js, vals, n, n)
end

@doc raw"""
Constructs the diagonal covariance matrix for noise vector $\epsilon$
for which $f = Bf + \epsilon$. 
See equation (10) of (Datta, A. Nearest neighbor sparse Cholesky
matrices in spatial statistics. 2022).
"""
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
	return Diagonal(vals)
end

struct NearestNeighbors
	k::Int
end

struct InvRoot{A}
	U::A
end

LinearAlgebra.logdet(A::InvRoot) = -2 * logdet(A.U) 

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
	return AbstractGPs.PosteriorGP(fx.f, (α=α, C=C, x=fx.x, δ=δ))
end

function API.approx_lml(nn::NearestNeighbors, fx::AbstractGPs.FiniteGP, y::AbstractVector)
  post = posterior(nn, fx, y)
  quadform = post.data.α' * post.data.δ
  ld = logdet(post.data.C)
  return -0.5 * ld -(length(y)/2) * log(2 * pi) - 0.5 * quadform
end

end
