function gauss_hermite_quadrature(
    y::AbstractVector,
    f_mean::AbstractVector,
    f_var::AbstractVector,
    lik;
    n_points=20
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see eg. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(n_points)
    # size(fs): (n_points, length(y))
    fs = √2 * .√f_var .* transpose(xs) .+ f_mean
    lls = loglikelihood.(lik.(fs), y)
    return (1/√π) * lls * ws
end

ChainRulesCore.@non_differentiable gausshermite(n)
