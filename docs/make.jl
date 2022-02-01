using ApproximateGPs
using Documenter

### Process examples
# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# The following function returns a script for instantiating the package
# environment of an example
# It ensures that the version of the package in which module `mod` is defined
# (here `mod` will be ApproximateGPs loaded above) is added to the environment.
# Thus the docs and the examples will use the same version of this package.
# There are two special cases:
# - If the script is executed by a GH action triggered by a new tag (i.e., release)
#   in the package repo, then this version of the package will be downloaded and installed
#   instead of checking out the local path
# - If the script is executed by a GH action triggered by a new commit in the `devbranch`
#   (by default `master`) in the package repo, then this revision of the package will be
#   downloaded and installed instead of checking out the local path
# This ensures that in these two cases the resulting Manifest.toml files do not fix the
# local path of any package, and hence can be used to reproduce the package environment in
# a clean and reproducible way.
function instantiate_script(mod; org, name=string(nameof(mod)), devbranch="master")
    github_repo = get(ENV, "GITHUB_REPOSITORY", "")
    github_event_name = get(ENV, "GITHUB_EVENT_NAME", "")

    repo = org * "/" * name * ".jl"
    if github_repo == repo && github_event_name == "push"
        github_ref = get(ENV, "GITHUB_REF", "")
        match_tag = match(r"^refs\/tags\/(.*)$", github_ref)
        if match_tag !== nothing
            # tagged release
            tag_nobuild = Documenter.version_tag_strip_build(match_tag.captures[1])
            if tag_nobuild !== nothing
                @info "Run examples with $name version $tag_nobuild"
                return """
using Pkg
Pkg.add(PackageSpec(; name="$name", version="$version"))
Pkg.instantiate()
"""
            end
        else
            # no release tag
            match_branch = match(r"^refs\/heads\/(.*)$", github_ref)
            if match_branch !== nothing && string(m.captures[1]) == devbranch
                sha = get(ENV, "GITHUB_SHA", nothing)
                if sha !== nothing
                    @info "Run examples with $name commit $sha"
                    return """
using Pkg
Pkg.add(PackageSpec(; name="$name", rev="$sha"))
Pkg.instantiate()
"""
                end
            end
        end
    end

    # Default: Use local path of provided module
    pkgdir_mod = pkgdir(mod)
    @info "Run examples with $name, local path $pkgdir_mod"
    return """
using Pkg
Pkg.develop(PackageSpec(; path="$pkgdir_mod"))
Pkg.instantiate()
"""
end

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
examples = filter!(isdir, readdir(joinpath(@__DIR__, "..", "examples"); join=true))
let script = instantiate_script(ApproximateGPs; org="JuliaGaussianProcesses")
    for example in examples
        if !success(`$(Base.julia_cmd()) --project=$example -e $script`)
            error(
                "project environment of example ",
                basename(example),
                " could not be instantiated",
            )
        end
    end
end
# Run examples asynchronously
processes = let literatejl = joinpath(@__DIR__, "literate.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $literatejl $(basename(example)) $EXAMPLES_OUT`;
                stdin=devnull,
                stdout=devnull,
                stderr=stderr,
            );
            wait=false,
        )::Base.Process
    end
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")

### Build documentation

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
        "API" => "api.md",
        "Examples" =>
            map(filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT))) do x
                return joinpath("examples", x)
            end,
    ],
    strict=true,
    checkdocs=:exports,
    doctestfilters=[
        r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
        r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
        r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/ApproximateGPs.jl.git", push_preview=true
)
