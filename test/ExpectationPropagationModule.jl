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
        Z_gh, q_gh = ExpectationPropagationModule.moment_match(cav_i, lik_eval_i; n_points=100)
        Z_quad, q_quad = moment_match_quadgk(cav_i, lik_eval_i)
        @test Z_gh ≈ Z_quad
        @test mean(q_gh) ≈ mean(q_quad)
        @test std(q_gh) ≈ std(q_quad)
    end

    @testset "predictions" begin
        approx = ApproximateGPs.ExpectationPropagation(; n_gh=500)
        ApproximateGPs.TestUtils.test_approximation_predictions(approx)
    end
end
