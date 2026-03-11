module API

export approx_lml  # this was moved into AbstractGPs.approx_log_evidence, see https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/pull/361

@deprecate approx_lml(approx, lfx, ys) approx_log_evidence(approx, lfx, ys)

end
