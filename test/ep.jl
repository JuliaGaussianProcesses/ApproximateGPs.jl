@testset "Expectation Propagation" begin
    @testset "moment_match" begin
        function moment_match_quadgk(cav_i::UnivariateDistribution, lik_eval_i)
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

        cav_i = Normal(0.8231, 3.213622)  # random numbers
        lik_eval_i = f -> pdf(Bernoulli(logistic(f)), true)
        Z_gh, q_gh = ApproximateGPs.moment_match(cav_i, lik_eval_i; n_points=100)
        Z_quad, q_quad = moment_match_quadgk(cav_i, lik_eval_i)
        @test Z_gh ≈ Z_quad
        @test mean(q_gh) ≈ mean(q_quad)
        @test std(q_gh) ≈ std(q_quad)
    end

    @testset "predictions" begin
        approx = ExpectationPropagation(; n_gh=500)
        test_approximation_predictions(approx)
    end

    @testset "approx_lml" begin
        k = 2SqExponentialKernel()
        x = [0.0, 1.0, 1.3]
        y = [0, 1, 0]

        ep = ExpectationPropagation()
        Φ(f) = cdf(Normal(), f)  # normcdf
        dist_y_given_f(f) = Bernoulli(Φ(f))
        lf = LatentGP(GP(k), dist_y_given_f, 1e-8)
        lfx = lf(x)

        function ep_probit_lml(ep, lfx, y)
            K = cov(lfx.fx)  # K = kernelmatrix(k, x)
            N = length(y)
            @assert size(K) == (N, N)
            ep_problem = ApproximateGPs.EPProblem(lfx.lik, y, K; ep)
            ep_state = ApproximateGPs.EPState(ep_problem)
            ep_state = ApproximateGPs.ep_outer_loop(ep_problem, ep_state)  # EP to convergence
            cavμ = zeros(N)
            cavσ² = zeros(N)
            μ̃ = zeros(N)
            σ̃² = zeros(N)
            for i in 1:N
                site_data = ApproximateGPs.ep_single_site_update(ep_problem, ep_state, i)
                μ̃[i] = mean(site_data.q)
                σ̃²[i] = var(site_data.q)
                cavμ[i] = mean(site_data.cav)
                cavσ²[i] = var(site_data.cav)
            end
            Σ̃ = Diagonal(σ̃²)
            term1 = -0.5logdet(K + Σ̃)
            term2 = -0.5μ̃' * ((K + Σ̃) \ μ̃)
            term3 = sum(@. log(Φ((y * cavμ) / √(1 + cavσ²))))
            term4 = 0.5sum(@. log(cavσ² + σ̃²))
            term5 = sum(@. (cavμ - μ̃)^2 / (cavσ² + σ̃²)) / 2
            @info "probit RW"
            @show term1
            @show term2
            @show term3
            @show term4
            @show term5
            @show term3 + term5 + term4
            return term1 + term2 + term3 + term4 + term5
        end
        @show ep_probit_lml(ep, lfx, y)
        @show approx_lml(ep, lfx, y)
        @show approx_lml(LaplaceApproximation(), lfx, y)
    end
end
