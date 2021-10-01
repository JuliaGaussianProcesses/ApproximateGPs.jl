struct ExpectationPropagation
    maxiter::Int
    epsilon::Float64
    n_gh::Int
end

function ExpectationPropagation(; maxiter=100, epsilon=1e-6, n_gh=150)
    return ExpectationPropagation(maxiter, epsilon, n_gh)
end

function AbstractGPs.posterior(ep::ExpectationPropagation, lfx::LatentFiniteGP, ys)
    ep_state = ep_inference(ep, lfx, ys)
    # TODO: it seems a bit weird to piggyback on SVGP here...
    return posterior(SVGP(lfx.fx, ep_state.q))
end

function ep_inference(ep::ExpectationPropagation, lfx::LatentFiniteGP, ys)
    fx = lfx.fx
    @assert mean(fx) == zero(mean(fx))  # might work with non-zero prior mean but not checked
    @assert length(ys) == length(fx)  # ExpectationPropagation currently does not support multi-latent likelihoods
    dist_y_given_f = lfx.lik
    K = cov(fx)

    return ep_inference(dist_y_given_f, ys, K; ep)
end

function ep_inference(dist_y_given_f, ys, K; ep=nothing)
    ep_problem = EPProblem(dist_y_given_f, ys, K; ep)
    ep_state = EPState(ep_problem)
    return ep_outer_loop(ep_problem, ep_state)
end

function EPProblem(ep::ExpectationPropagation, p::MvNormal, lik_evals::AbstractVector)
    return (; p, lik_evals, ep)
end

function EPProblem(dist_y_given_f, ys, K; ep=nothing)
    f_prior = MvNormal(K)
    lik_evals = [f -> pdf(dist_y_given_f(f), y) for y in ys]
    return EPProblem(ep, f_prior, lik_evals)
end

function EPState(q::MvNormal, sites::AbstractVector)
    return (; q, sites)
end

function EPState(ep_problem)
    N = length(ep_problem.lik_evals)
    # TODO- manually keep track of canonical parameters and initialize precision to 0
    sites = [(; q=NormalCanon(0.0, 1e-10)) for _ in 1:N]
    q = ep_problem.p
    return EPState(q, sites)
end

function ep_approx_posterior(prior, sites::AbstractVector)
    canon_site_dists = [convert(NormalCanon, t.q) for t in sites]
    potentials = [q.η for q in canon_site_dists]
    precisions = [q.λ for q in canon_site_dists]
    ts_dist = MvNormalCanon(potentials, precisions)
    return mul_dist(prior, ts_dist)
end

function ep_outer_loop(ep_problem, ep_state; maxiter=ep_problem.ep.maxiter)
    for i in 1:maxiter
        @info "Outer loop iteration $i"
        new_state = ep_loop_over_sites(ep_problem, ep_state)
        if ep_converged(ep_state.sites, new_state.sites; epsilon=ep_problem.ep.epsilon)
            @info "converged"
            break
        else
            ep_state = new_state
        end
    end
    return ep_state
end

function ep_converged(old_sites, new_sites; epsilon=1e-6)
    # TODO improve convergence check
    diff1 = [(t_old.q.η - t_new.q.η)^2 for (t_old, t_new) in zip(old_sites, new_sites)]
    diff2 = [(t_old.q.λ - t_new.q.λ)^2 for (t_old, t_new) in zip(old_sites, new_sites)]
    return mean(diff1) < epsilon && mean(diff2) < epsilon
end

function ep_loop_over_sites(ep_problem, ep_state)
    # TODO: randomize order of updates: make configurable?
    for i in randperm(length(ep_problem.lik_evals))
        @info "  Inner loop iteration $i"
        new_t = ep_single_site_update(ep_problem, ep_state, i)

        # TODO: rank-1 update
        new_sites = deepcopy(ep_state.sites)
        new_sites[i] = (; q=new_t)
        new_q = meanform(ep_approx_posterior(ep_problem.p, new_sites))
        ep_state = EPState(new_q, new_sites)
    end
    return ep_state
end

function ep_single_site_update(ep_problem, ep_state, i::Int)
    q_fi = ith_marginal(ep_state.q, i)
    alik_i = epsite_dist(ep_state.sites[i])
    cav_i = div_dist(q_fi, alik_i)
    qhat_i = moment_match(cav_i, ep_problem.lik_evals[i]; n_points=ep_problem.ep.n_gh)
    new_t = div_dist(qhat_i.q, cav_i)
    return new_t
end

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

function moment_match(cav_i::Union{Normal,NormalCanon}, lik_eval_i; n_points=150)
    # TODO: combine with expected_loglik / move into GPLikelihoods
    xs, ws = gausshermite(n_points)
    fs = √2 * std(cav_i) * xs .+ mean(cav_i)
    scale = (1 / √π)
    lik_ws = lik_eval_i.(fs) .* ws
    fs_lik_ws = fs .* lik_ws
    fs²_lik_ws = fs .* fs_lik_ws
    m0 = scale * sum(lik_ws)
    m1 = scale * sum(fs_lik_ws)
    m2 = scale * sum(fs²_lik_ws)
    matched_Z = m0
    matched_mean = m1 / m0
    matched_var = m2 / m0 - matched_mean^2
    return (; Z=matched_Z, q=Normal(matched_mean, sqrt(matched_var)))
end
