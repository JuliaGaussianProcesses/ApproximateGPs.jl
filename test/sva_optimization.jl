# Similar to examples/b-classification/script.jl

@testset "sparse variational optimization" begin
Random.seed!(1234)

k_true = [30.0, 1.5]
kernel_true = k_true[1] * (SqExponentialKernel() ∘ ScaleTransform(k_true[2]))

jitter = 1e-8  # for numeric stability
lgp = LatentGP(GP(kernel_true), BernoulliLikelihood(), jitter)
x_true = 0:0.02:6
f_true, y_true = rand(lgp(x_true))
  
μ = mean.(lgp.lik.(f_true))
plot(x_true, μ; seriescolor="red", label="")

N = 30  # The number of training points
mask = sample(1:length(x_true), N; replace=false, ordered=true)  # Subsample some input locations
x, y = x_true[mask], y_true[mask]

M = 15  # number of inducing points
raw_initial_params = (
    k=(var=positive(rand()), precision=positive(rand())),
    z=bounded.(range(0.1, 5.9; length=M), 0.0, 6.0),  # constrain z to simplify optimisation
    m=zeros(M),
    A=positive_definite(Matrix{Float64}(I, M, M)),
);

flat_init_params, unflatten = ParameterHandling.flatten(raw_initial_params)
unpack = ParameterHandling.value ∘ unflatten;

lik = BernoulliLikelihood()
jitter = 1e-3  # added to aid numerical stability

function build_SVGP(params::NamedTuple)
    kernel = params.k.var * (SqExponentialKernel() ∘ ScaleTransform(params.k.precision))
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(params.m, params.A)
    fz = f(params.z).fx
    return SparseVariationalApproximation(fz, q), f
end

function loss(params::NamedTuple)
    svgp, f = build_SVGP(params)
    fx = f(x)
    return -elbo(svgp, fx, y)
end;

# Optimise the parameters using LBFGS.

opt = optimize(
    loss ∘ unpack,
    θ -> only(Zygote.gradient(loss ∘ unpack, θ)),
    flat_init_params,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=4_000);
    inplace=false,
)

# Finally, build the optimised SVGP model, and sample some functions to see if
# they are close to the true function.

final_params = unpack(opt.minimizer)

# Check whether optimized variational covariance matrix is non-diagonal
@test final_params.A != Diagonal(diag(final_params.A))
end
