export ith_marginal, mul_dist, div_dist, moment_match, ep_approx_posterior, epsite_pdf

function ith_marginal(d::Union{MvNormal,MvNormalCanon}, i::Int)
    m = mean(d)
    v = var(d)
    return Normal(m[i], sqrt(v[i]))
end
function mul_dist(a::NormalCanon, b::NormalCanon)
    # NormalCanon
    #     η::T       # σ^(-2) * μ
    #     λ::T       # σ^(-2)
    etaAmulB = a.η + b.η
    lambdaAmulB = a.λ + b.λ
    return NormalCanon(etaAmulB, lambdaAmulB)
end

mul_dist(a, b) = mul_dist(convert(NormalCanon, a), convert(NormalCanon, b))

function mul_dist(a::MvNormalCanon, b::MvNormalCanon)
    # MvNormalCanon
    #    h::V    # potential vector, i.e. inv(Σ) * μ
    #    J::P    # precision matrix, i.e. inv(Σ)
    hAmulB = a.h + b.h
    JAmulB = a.J + b.J
    return MvNormalCanon(hAmulB, JAmulB)
end

mul_dist(a::MvNormal, b) = mul_dist(canonform(a), b)

function div_dist(a::NormalCanon, b::NormalCanon)
    # NormalCanon
    #     η::T       # σ^(-2) * μ
    #     λ::T       # σ^(-2)
    etaAdivB = a.η - b.η
    lambdaAdivB = a.λ - b.λ
    return NormalCanon(etaAdivB, lambdaAdivB)
end

div_dist(a::Normal, b) = div_dist(convert(NormalCanon, a), b)
div_dist(a, b::Normal) = div_dist(a, convert(NormalCanon, b))

#function EPSite(Z, m, s2)
#    return (; Z, m, s2)
#end
#
#function epsite_dist(site)
#    return Normal(site.m, sqrt(site.s2))
#end

epsite_dist(site) = site.q

function epsite_pdf(site, f)
    return site.Z * pdf(epsite_dist(site), f)
end

function moment_match(cav_i::UnivariateDistribution, lik_eval_i)
    lower = mean(cav_i) - 20 * std(cav_i)
    upper = mean(cav_i) + 20 * std(cav_i)
    m0, _ = quadgk(f -> pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    m1, _ = quadgk(f -> f * pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    m2, _ = quadgk(f -> f^2 * pdf(cav_i, f) * lik_eval_i(f), lower, upper)
    matched_Z = m0
    matched_mean = m1 / m0
    matched_var = m2 / m0 - matched_mean^2
    return (; Z=matched_Z, q=Normal(matched_mean, sqrt(matched_var)))
end

function ep_approx_posterior(prior, sites::AbstractVector)
    canon_site_dists = [convert(NormalCanon, t.q) for t in sites]
    potentials = [q.η for q in canon_site_dists]
    precisions = [q.λ for q in canon_site_dists]
    ts_dist = MvNormalCanon(potentials, precisions)
    return mul_dist(prior, ts_dist)
end

function EPProblem(p::MvNormal, lik_evals::AbstractVector)
    return (; p, lik_evals)
end

function EPState(q::MvNormal, sites::AbstractVector)
    return (; q, sites)
end

function ep_single_site_update(ep_problem, ep_state, i::Int)
    q_fi = ith_marginal(ep_state.q, i)
    alik_i = epsite_dist(ep_state.sites[i].q)
    cav_i = div_dist(q_fi, alik_i)
    qhat_i = moment_match(cav_i, ep_problem.lik_evals[i])
    return new_t = div_dist(qhat_i.q, cav_i)
    #delta_eta = 
    #new_q = rank_one_update(ep_state.q, 

    #new_q = 
    #return EPState(new_q, new_sites)
end

function ep_step!(ep_tuple)
    return "work in progress"
end

function EPResult(results)
    return (; results)
end

function ep_steps(dist_y_given_f, f_prior, y; maxiter=100)
    f = mean(f_prior)
    @assert f == zero(f)  # might work with non-zero prior mean but not checked
    converged = false
    res_array = []
    for i in 1:maxiter
        results = ep_step!(f, dist_y_given_f, f_prior, y)
        push!(res_array, EPResult(results))
        if isapprox(f, results.fnew)
            break  # converged
        else
            f = results.fnew
        end
    end
    return res_array
end
