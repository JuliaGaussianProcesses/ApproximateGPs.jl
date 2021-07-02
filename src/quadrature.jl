
function gauss_hermite_quadrature(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik::BernoulliLikelihood;
    n_points=20
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see eg. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(n_points)
    fs = √2 * .√f_var' .* xs .+ f_mean'
    lls = loglikelihood.(lik.(fs), y')
    return ((1/√π) * ws'lls)'
end

ChainRulesCore.@non_differentiable gausshermite(n)
