### Process examples
using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl")) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

using ApproximateGPs

JuliaGPsDocs.generate_examples(ApproximateGPs)

### Build documentation
using Documenter

# Doctest setup
DocMeta.setdocmeta!(
    ApproximateGPs,
    :DocTestSetup,
    quote
        using ApproximateGPs
    end;  # we have to load all packages used (implicitly) within jldoctest blocks in the API docstrings
    recursive=true,
)

makedocs(;
    sitename="ApproximateGPs.jl",
    format=Documenter.HTML(),
    modules=[ApproximateGPs],
    pages=[
        "Home" => "index.md",
        "userguide.md",
        "API" => joinpath.(Ref("api"), ["index.md", "sparsevariational.md", "laplace.md"]),
        "Examples" => JuliaGPsDocs.find_generated_examples(ApproximateGPs),
    ],
    strict=true,
    checkdocs=:exports,
    doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/ApproximateGPs.jl.git", push_preview=true
)
