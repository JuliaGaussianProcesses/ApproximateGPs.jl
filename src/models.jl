const default_jitter = 1e-6

struct SVGPModel{Tlik}
    kernel_func # function to construct the kernel from `k`
    lik::Tlik   # the likelihood function
    jitter      # the jitter added to covariance matrices
    
    ## Trainable parameters
    k           # kernel parameters
    m           # variational mean
    A           # variational covariance (sqrt)
    z           # inducing points
end

@functor SVGPModel (k, m, A, z,)

function SVGPModel(
    kernel_func,
    kernel_params,
    inducing_inputs;
    q_μ::Union{Vector, Nothing}=nothing,
    q_Σ_sqrt::Union{Matrix, Nothing}=nothing,
    jitter=default_jitter,
    likelihood=GaussianLikelihood(jitter)
)
    m, A = _init_variational_params(q_μ, q_Σ_sqrt, inducing_inputs)
    return SVGPModel(
        kernel_func,
        likelihood,
        jitter,
        kernel_params,
        m,
        A,
        inducing_inputs
    )
end

function (m::SVGPModel{GaussianLikelihood})(x)
    S = PDMat(Cholesky(LowerTriangular(A)))
    q = MvNormal(m.m, S)
    kernel = m.kernel_func(m.k)
    f = GP(kernel)
    fx = f(x, m.lik.σ²)
    fu = f(m.z, m.jitter)
    return fx, fu, q
end

function (m::SVGPModel)(x)
    S = PDMat(Cholesky(LowerTriangular(A)))
    q = MvNormal(m.m, S)
    kernel = m.kernel_func(m.k)
    f = LatentGP(GP(kernel), m.lik, m.jitter)
    fx = f(x)
    fu = f(m.z).fx
    return fx, fu, q
end

function loss(m::SVGPModel, x, y; n_data=length(y))
    return -elbo(m, x, y, n_data)
end

function elbo(m::SVGPModel, x, y; n_data=length(y))
    fx, fu, q = m(x)
    return SparseGPs.elbo(fx, y, fu, q; n_data)
end

function _init_variational_params(n)
    m = zeros(n)
    A = Matrix{Float64}(I, n, n)
    return m, A
end

function _init_variational_params(q_μ, q_Σ_sqrt, z)
    n = length(z)
    if q_μ === nothing
        q_μ = zeros(n)
    end
    if q_Σ_sqrt === nothing
        q_Σ_sqrt = Matrix{Float64}(I, n, n)
    end
    return q_μ, q_Σ_sqrt
end
