@testset "equivalences" begin
    rng, N = MersenneTwister(654321), 20
    x = rand(rng, N)
    y = sin.(x) + 0.9 * cos.(x * 1.6) + 0.4 * rand(rng, N)

    z = copy(x) # Set inducing inputs == training inputs
    
end
