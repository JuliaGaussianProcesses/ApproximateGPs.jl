function pathwise_sample(rng::Random.AbstractRNG, f::ApproxPosteriorGP, weight_space_approx) # TODO: num_samples
    prior_approx = weight_space_approx(f.prior)
    prior_sample = rand(rng, prior_approx)

    z = inducing_points(f)
    q_u = _get_q_u(f)

    u = rand(rng, q_u)
    v = cov(f, z) \ (u - prior_sample(z))

    posterior_sample(x) = prior_sample(x) + cov(f, x, z) * v

    return posterior_sample
end

# Methods to get the explicit variational distribution over inducing points q(u)
function _get_q_u(f::ApproxPosteriorGP{<:SparseVariationalApproximation{NonCentered}})
    # u = Lε + μ where LLᵀ = cov(fz) and μ = mean(fz)
    # q(ε) = N(m, S)
    # => q(u) = N(Lm + μ, LSLᵀ)
    L, μ = chol_lower(_chol_cov(f.approx.fz)), mean(f.approx.fz)
    m, S = mean(f.approx.q), _chol_cov(f.approx.q)
    return MvNormal(L * m + μ, Xt_A_X(S, L'))
end
_get_q_u(f::ApproxPosteriorGP{<:SparseVariationalApproximation{Centered}}) = f.approx.q

function _get_q_u(f::ApproxPosteriorGP{<:VFE})
    # q(u) = N(m, S)
    # q(f_k) = N(μ_k, Σ_k)  (the predictive distribution at test inputs k)
    # μ_k = mean(k) + K_kz * K_zz⁻¹ * m
    # where: K_kz = cov(f.prior, k, z)
    # implemented as: μ_k = mean(k) + K_kz * α
    # => m = K_zz * α
    # Σ_k = K_kk - (K_kz * K_zz⁻¹ * K_zk) + (K_kz * K_zz⁻¹ * S * K_zz⁻¹ * K_zk)
    # interested in the last term to get S
    # implemented as: Aᵀ * Λ_ε⁻¹ * A
    # where: A = U⁻ᵀ * K_zk
    # UᵀU = K_zz
    # so, Λ_ε⁻¹ = U⁻ᵀ * S * U
    # => S = Uᵀ * Λ_ε⁻¹ * U
    # see https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/ eqns (8) & (9)
    U = f.data.U
    m = U'U * f.data.α
    S = Xt_invA_X(f.data.Λ_ε, U)
    return MvNormal(m, S)
end
