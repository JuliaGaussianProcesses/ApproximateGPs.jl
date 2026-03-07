### Process examples
using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl")) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

using ApproximateGPs

JuliaGPsDocs.generate_examples(ApproximateGPs; ntasks=1)

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
    checkdocs=:exports,
    doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
)
