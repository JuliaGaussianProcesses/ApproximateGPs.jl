# Create a default kernel from two parameters k[1] and k[2]
make_kernel(k) = softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))

# Computes the optimal closed form solution for the variational posterior
# q(u) (e.g. # https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
# equations (11) & (12)).
function exact_variational_posterior(fu, fx, y)
    σ² = fx.Σy[1]
    Kuf = cov(fu, fx)
    Kuu = Symmetric(cov(fu))
    Σ = (Symmetric(cov(fu) + (1/σ²) * Kuf * Kuf'))
    m = ((1/σ²)*Kuu* (Σ\Kuf)) * y
    S = Symmetric(Kuu * (Σ \ Kuu))
    return MvNormal(m, S)
end
