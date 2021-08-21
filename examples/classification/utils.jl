# A quick and dirty positive definite matrix type for ParameterHandling.jl

struct PDMatrix{TA}
    A::TA
end

function pdmatrix(A::AbstractMatrix)
    return PDMatrix(A)
end

function ParameterHandling.value(P::PDMatrix)
    A = copy(P.A)
    return A'A
end

function ParameterHandling.flatten(::Type{T}, P::PDMatrix) where {T}
    v, unflatten_to_Array = flatten(T, P.A)
    function unflatten_PDmatrix(v_new::Vector{T})
        A = unflatten_to_Array(v_new)
        return PDMatrix(A)
    end
    return v, unflatten_PDmatrix
end
