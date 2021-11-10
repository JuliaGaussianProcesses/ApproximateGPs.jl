@testset "utils" begin
    # Copied from https://github.com/JuliaLang/julia/pull/39352/files
    @testset "constructing a Cholesky factor from a triangular matrix" begin
        A = [1.0 2.0; 3.0 4.0]
        let
            U = UpperTriangular(A)
            C = Cholesky(U)
            @test C isa Cholesky{Float64}
            @test C.U == U
            @test C.L == U'
        end
        let
            L = LowerTriangular(A)
            C = Cholesky(L)
            @test C isa Cholesky{Float64}
            @test C.L == L
            @test C.U == L'
        end
end
