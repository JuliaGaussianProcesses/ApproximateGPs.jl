# Retrieve name of example and output directory
if length(ARGS) != 2
    error("please specify the name of the example and the output directory")
end
const EXAMPLE = ARGS[1]
const OUTDIR = ARGS[2]

# Activate environment
# Note that each example's Project.toml must include Literate as a dependency
using Pkg: Pkg

using InteractiveUtils
const EXAMPLEPATH = joinpath(@__DIR__, "..", "examples", EXAMPLE)
Pkg.activate(EXAMPLEPATH)
Pkg.instantiate()
pkg_status = sprint() do io
    Pkg.status(; io=io)
end

using Literate: Literate

const MANIFEST_OUT = joinpath(EXAMPLE, "Manifest.toml")
mkpath(joinpath(OUTDIR, EXAMPLE))
cp(joinpath(EXAMPLEPATH, "Manifest.toml"), joinpath(OUTDIR, MANIFEST_OUT); force=true)

""" adapted from HttpCommon.jl """
function escapeHTML(i::String)
    # Refer to http://stackoverflow.com/a/7382028/3822752 for spec. links
    o = replace(i, "&" => "&amp;")
    o = replace(o, "\"" => "&quot;")
    o = replace(o, "'" => "&#39;")
    o = replace(o, "<" => "&lt;")
    o = replace(o, ">" => "&gt;")
    return o
end

function preprocess(content)
    # Add link to nbviewer below the first heading of level 1
    sub = SubstitutionString(
        """
#md # ```@meta
#md # EditURL = "@__REPO_ROOT_URL__/examples/@__NAME__/script.jl"
#md # ```
#md #
\\0
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)
#md #
# *You are seeing the
#md # HTML output generated by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and
#nb # notebook output generated by
# [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the
# [Julia source file](@__REPO_ROOT_URL__/examples/@__NAME__/script.jl).
#md # The corresponding notebook can be viewed in [nbviewer](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb).*
#nb # The rendered HTML can be viewed [in the docs](https://juliagaussianprocesses.github.io/ApproximateGPs.jl/dev/examples/@__NAME__/).*
#
# ---
#
        """,
    )
    content = replace(content, r"^# # [^\n]*"m => sub; count=1)

    # remove VSCode `##` block delimiter lines
    content = replace(content, r"^##$."ms => "")

    """ The regex adds "# " at the beginning of each line; chomp removes trailing newlines """
    literate_format(s) = chomp(replace(s, r"^"m => "# "))

    # <details></details> seems to be buggy in the notebook, so is avoided for now
    info_footer = """
    #md # ```@raw html
    # <hr />
    # <details>
    # <summary><strong>Package and system information (click to expand)</strong></summary>
    # <h4>Package versions</h4>
    # <pre>
    $(literate_format(escapeHTML(pkg_status)))
    # </pre>
    # <h4>System information</h4>
    # <pre>
    $(literate_format(escapeHTML(sprint(InteractiveUtils.versioninfo))))
    # </pre>
    # <h4>Manifest</h4>
    # To reproduce this notebook's package environment, you can
    #nb # <a href="$(MANIFEST_OUT)">
    #md # <a href="../$(MANIFEST_OUT)">
    # download the full Manifest.toml</a>.
    # </details>
    #md # ```
    """

    return content * info_footer
end

function md_postprocess(content)
    return replace(content, r"[\n]nothing #hide$"m => "")
end

# Convert to markdown and notebook
const SCRIPTJL = joinpath(EXAMPLEPATH, "script.jl")
Literate.markdown(
    SCRIPTJL,
    OUTDIR;
    name=EXAMPLE,
    documenter=true,
    execute=true,
    preprocess=preprocess,
    postprocess=md_postprocess,
)
Literate.notebook(
    SCRIPTJL, OUTDIR; name=EXAMPLE, documenter=true, execute=true, preprocess=preprocess
)
