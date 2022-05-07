using LinearAlgebra
using Random
using Test

using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
using Flux: Flux
using IterTools
using LogExpFunctions: softplus
using Optim
using PDMats
using Zygote

using AbstractGPs
using ApproximateGPs
using ApproximateGPs: SparseVariationalApproximationModule, LaplaceApproximationModule

# Writing tests:
# 1. The file structure of the test should match precisely the file structure of src.
#   Amongst other things, this means that there should be exactly 1 test file per src file.
#   This makes it trivially easy for someone to find the tests associated with a particular
#   src file.
# 2. A consequence of 1 is that there should be exactly 1 test file per src file.
# 3. A test file called foo.jl should have the structure:
#   @testset "foo" begin
#       code
#   end
#
#   Note that the testset is called `foo`, not `foo.jl`. Use whatever testset structure
#   seems appropriate within a given file. eg. if multiple types / functions are defined in
#   a particular source file, you might want multiple testsets in the test file.
# 4. Each directory should have its own testset, in which each test file is `include`d.
# 5. Each test file should create its own state, and shouldn't rely on state defined in
#   other test files. If two test files are similar enough to share state, perhaps the
#   corresponding source code should be in the same file.
# 6. If you write a src file without any tests, create a corresponding test file with the
#   usual structure, but without any tests.
# 7. Explicitly create a new random number generate for _at_ _least_ each new test file, and
#   use it whenever generating randomness. This ensures complete control over random number
#   generation and makes it clear what randomness depends on other randomness.
# 8. All `using` statements should appear in runtests.jl.
# 9. List out all test files explicitly (eg. don't loop over them). This makes it easy to
#   disable tests by simply commenting them out, and makes it very clear which tests are not
#   currently being run.
# 10. If utility functionality is required, it should be placed in `src/test_utils.jl` so
#   that other tests can benefit from it.
@info "Packages Loaded"

include("test_utils.jl")

@testset "ApproximateGPs" begin
    include("SparseVariationalApproximationModule.jl")
    println(" ")
    @info "Ran sva tests"

    include("LaplaceApproximationModule.jl")
    println(" ")
    @info "Ran laplace tests"
end
