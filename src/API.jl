module API

export approx_lml  # TODO move to AbstractGPs, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues/221

@deprecate approx_lml(approx, lfx::LatentFiniteGP, ys) approx_log_evidence(approx, lfx, ys)

end
