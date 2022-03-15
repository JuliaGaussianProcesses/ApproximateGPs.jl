# Create a default kernel from two parameters k[1] and k[2]
make_kernel(k) = softplus(k[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(k[2])))
