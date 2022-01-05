using DelimitedFiles
fn = "CASP.csv"  # protein dataset via bayesian_benchmarks
dat = readdlm(fn, ',', Float64, '\n'; skipstart=1)

Y = dat[:, 1]
X = dat[:, 2:end]
D = size(X, 2)

using Statistics
Xs = (X .- mean(X, dims=1)) ./ std(X, dims=1)
Ys = (Y .- mean(Y)) ./ std(Y)

function prep_train_data(Ntrain)
    Xtrain = Xs[1:Ntrain, :]
    Ytrain = Ys[1:Ntrain]
    return (Xtrain, Ytrain)
end

using Random

function prep_Z(Xtrain, M; seed=12345)
    Random.seed!(seed)
    Ntrain = size(Xtrain, 1)
    idxZ = randperm(Ntrain)[1:M]
    Z = Xtrain[idxZ, :]
    return Z
end

using ApproximateGPs
using ParameterHandling
using LinearAlgebra: I
using Distributions: MvNormal

function prep_initial_params(Z)
    M = size(Z, 1)
    raw_initial_params = (
        k=(var=positive(1.0), lengthscale=positive(1.0)),
        noisevar=positive(0.1),
        z=fixed(Z),
        m=zeros(M),
        A=positive_definite(Matrix{Float64}(I, M, M)),
    )
    flat_init_params, unpack = ParameterHandling.value_flatten(raw_initial_params)
    return flat_init_params, unpack
end

function build_SVGP(params::NamedTuple)
    kernel = params.k.var * with_lengthscale(SqExponentialKernel(), params.k.lengthscale)
    lik = GaussianLikelihood(params.noisevar)
    jitter = 1e-6
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(params.m, params.A)
    fz = f(RowVecs(params.z)).fx
    return SparseVariationalApproximation(fz, q), f
end

function make_loss(Xtrain, Ytrain)
    function loss(params::NamedTuple)
        svgp, f = build_SVGP(params)
        fx = f(RowVecs(Xtrain))
        return -elbo(svgp, fx, Ytrain)
    end

    return loss
end

#println(loss(unpack(flat_init_params)))

# Optimise the parameters using LBFGS.

using Optim
using Zygote

#maxiter = 100
#
#t0 = time()
#opt = optimize(
#    loss ∘ unpack,
#    θ -> only(Zygote.gradient(loss ∘ unpack, θ)),
#    flat_init_params,
#    LBFGS(;
#        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
#        linesearch=Optim.LineSearches.BackTracking(),
#    ),
#    Optim.Options(; iterations=maxiter);
#    inplace=false,
#)
#t1 = time()

function setup(Ntrain=5000, M=300)
    Xtrain, Ytrain = prep_train_data(Ntrain)
    Z = prep_Z(Xtrain, M)
    flat_init_params, unpack = prep_initial_params(Z)
    loss = make_loss(Xtrain, Ytrain)
    return loss ∘ unpack, flat_init_params
end
