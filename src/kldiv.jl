function KL(p::AbstractMvNormal, q::AbstractMvNormal)
    # This is the generic implementation for AbstractMvNormal, you might need to specialize for your type
    length(p) == length(q) || throw(
        DimensionMismatch(
            "Distributions p and q have different dimensions $(length(p)) and $(length(q))",
        ),
    )
    Σp, Σq = cov.((p, q))
    Δμ = mean(p) - mean(q)
    return 0.5 * (tr(Σq \ Σp) + dot(Δμ, Σq \ Δμ) - length(p) + logdet(Σq) - logdet(Σp))
end

function KL(p::MvNormal, q::MvNormal)
    # We use p.Σ and q.Σ to take the advantage that they are defined as PDMats objects
    length(p) == length(q) || throw(
        DimensionMismatch(
            "Distributions p and q have different dimensions $(length(p)) and $(length(q))",
        ),
    )
    return 0.5 * (
        tr(q.Σ \ p.Σ) + invquad(q.Σ, mean(p) - mean(q)) - length(p) + logdet(q.Σ) -
        logdet(p.Σ)
    )
end

kldivergence(p, q) = KL(p, q)

# opt-out of generic rrule for `Matrix` defined in ChainRules.jl.
# It's horrible that this code is needed. Given that it does, its ideal home is PDMats.jl.
# We're type-pirating for now.
ChainRulesCore.@opt_out ChainRulesCore.rrule(::Type{<:Matrix}, ::PDMats.PDMat)
