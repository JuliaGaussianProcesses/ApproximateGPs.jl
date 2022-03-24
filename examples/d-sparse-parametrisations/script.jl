# # The Various Pseudo-Point Approximation Parametrisations
#
# ### Note to the reader
# At the time of writing (March 2021) the best way to parametrise the approximate posterior
# remains a surprisingly active area of research.
# If you are reading this and feel that it has become outdated, or was incorrect in the
# first instance, it would be greatly appreciated if you could open an issue to discuss.
# 
#
# ## Introduction
#
# This example examines the various ways in which this package supports parametrising the
# approximate posterior when utilising sparse approximations.
# 
# All sparse (a.k.a. pseudo-point) approximations in this package utilise an approximate
# posterior over a GP ``f`` of the form
# ```math
# q(f) = q(\mathbf{u}) \, p(f_{\neq \mathbf{u}} | \mathbf{u}) 
# ```
# where samples from ``f`` are functions mapping ``\mathcal{X} \to \mathbb{R}``,
# ``\mathbf{u} := f(\mathbf{z})``, ``\mathbf{z} \in \mathcal{X}^M`` are the pseudo-inputs,
# and ``f_{\neq \mathbf{u}}`` denotes ``f`` at all indices other than those in
# ``\mathbf{z}``.[^Titsias]
# ``\mathbf{u} := q(f(\mathbf{z}))`` is generally restricted to be a multivariate Gaussian, to which end ApproximateGPs presently offers four parametrisations:
# 1. Centered ("Unwhitened"): ``q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{C})``, ``\quad \mathbf{m} \in \mathbb{R}^M`` and positive-definite ``\mathbf{C} \in \mathbb{R}^{M \times M}``,
# 1. Non-Centered ("Whitened"): ``q(\mathbf{u}) = \mathcal{N}(\mathbf{L} \mathbf{m}, \mathbf{L} \mathbf{C} \mathbf{T}^\top)``, ``\quad \mathbf{L} \mathbf{L}^\top = \text{cov}(\mathbf{u})``,
# 1. Pseudo-Observation: ``q(\mathbf{u}) \propto p(\mathbf{u}) \, \mathcal{N}(\hat{\mathbf{y}}; \mathbf{u}, \hat{\mathbf{S}})``, ``\quad \hat{\mathbf{y}} \in \mathbb{R}^M`` and positive-definite ``\hat{\mathbf{S}} \in \mathbb{R}^{M \times M}``,
# 1. Decoupled Pseudo-Observation: ``q(\mathbf{u}) \propto p(\mathbf{u}) \, \mathcal{N}(\hat{\mathbf{y}}; f(\mathbf{v}), \hat{\mathbf{S}})``, ``\quad \hat{\mathbf{y}} \in \mathbb{R}^R``, ``\hat{\mathbf{S}} \in \mathbb{R}^{R \times R}`` is positive-definite and diagonal, and ``\mathbf{v} \in \mathcal{X}^R``.
#
# The choice of parametrization can have a substantial impact on the time it takes for ELBO
# optimization to converge, and which parametrization is better in a particular situation is
# not generally obvious.
# That being said, the `NonCentered` parametrization often converges in fewer iterations
# than the `Centered`, and is widely used, so it is the default.
#
# For a general discussion around the centered vs non-centered, see e.g. [^Gorinova].
# For a GP-specific discussion, see e.g. section 3.4 of [^Paciorek].

# ## Setup

using AbstractGPs
using ApproximateGPs
using CairoMakie
using Distributions
using Images
using KernelFunctions
using LinearAlgebra
using Optim
using Random
using Zygote

# A simple GP with inputs on the reals.
f = GP(SEKernel());
N = 100;
x = range(-3.0, 3.0; length=N);

# Generate some observations.
Σ = Diagonal(fill(0.1, N));
y = rand(Xoshiro(123456), f(x, Σ));

# Use a handful of pseudo-points.
M = 10;
z = range(-3.5, 3.5; length=M);

# Other misc. constants that we'll need later:
x_pred = range(-5.0, 5.0; length=300);
jitter = 1e-9;

# ## The Relationship Between Parametrisations
#
# Much of the time, one can convert between the different parametrisations to obtain
# equivalent ``q(\mathbf{u})``, for a given set of hyperparameters.
# If it's unclear from the above how these parametrisations relate to one another, the
# following should help to crystalise the relationship.
#
# ### Centered vs Non-Centered
#
# Both the `Centered` and `NonCentered` parametrisations are specified by a mean vector `m`
# and covariance matrix `C`, but in slightly different ways.
# The `Centered` parametrisation interprets `m` and `C` as the mean and covariance of
# ``q(\mathbf{u})`` directly, while the `NonCentered` parametrisation inteprets them as the
# mean and covariance of the approximate posterior over
# `ε := cholesky(cov(u)).U' \ (u - mean(u))`.
#
# To see this, consider the following non-centered approximate posterior:
fz = f(z, jitter);
qu_non_centered = MvNormal(randn(M), Matrix{Float64}(I, M, M));
non_centered_approx = SparseVariationalApproximation(NonCentered(), fz, qu_non_centered);

# The equivalent centered parametrisation can be found by multiplying the parameters of
# `qu_non_centered` by the Cholesky factor of the prior covariance:
L = cholesky(Symmetric(cov(fz))).L;
qu_centered = MvNormal(L * mean(qu_non_centered), L * cov(qu_non_centered) * L');
centered_approx = SparseVariationalApproximation(Centered(), fz, qu_centered);

# We can gain some confidence that they're actually the same by querying the approximate
# posterior statistics at some new locations:
q_non_centered = posterior(non_centered_approx)
q_centered = posterior(centered_approx)
@assert mean(q_non_centered(x_pred)) ≈ mean(q_centered(x_pred))
@assert cov(q_non_centered(x_pred)) ≈ cov(q_centered(x_pred))


# ### Pseudo-Observation vs Centered
#
# The relationship between these two parametrisations is only slightly more complicated.
# Consider the following pseudo-observation parametrisation of the approximate posterior:
ŷ = randn(M);
Ŝ = Matrix{Float64}(I, M, M);
pseudo_obs_approx = PseudoObsSparseVariationalApproximation(f, z, Ŝ, ŷ);
q_pseudo_obs = posterior(pseudo_obs_approx);

# The corresponding centered approximation is given via the usual Gaussian conditioning
# formulae:
C = cov(fz);
C_centered = C - C * (cholesky(Symmetric(C + Ŝ)) \ C);
m_centered = mean(fz) + C / cholesky(Symmetric(C + Ŝ)) * (ŷ - mean(fz));
qu_centered = MvNormal(m_centered, Symmetric(C_centered));
centered_approx = SparseVariationalApproximation(Centered(), fz, qu_centered);
q_centered = posterior(centered_approx);

# Again, we can gain some confidence that they're the same by comparing the posterior
# marginal statistics.
@assert mean(q_pseudo_obs(x_pred)) ≈ mean(q_centered(x_pred))
@assert cov(q_pseudo_obs(x_pred)) ≈ cov(q_centered(x_pred))

# While it's always possible to find an approximation using the centered parametrisation
# which is equivalent to a given pseudo-observation parametrisation, the converse is not
# true.
# That is, for a given `C = cov(fz)` and particular choice of covariance matrix `Ĉ` in a
# centered parametrisation, it may not be the case that there exists a positive-definite
# pseudo-observation covariance matrix `Ŝ` such that ``\hat{C} = C - C (C + \hat{S})^{-1} C``.
#
# However, ths is not necessarily a problem: if the likelihood used in the model is
# log-concave then the optimal choice for `Ĉ` can always be represented using this
# pseudo-observation parametrisation.
# Even when this is not the case, it is not guaruanteed to be the case that the optimal
# choice for `q(u)` lives outside of the family of distributions which can be expressed
# within the pseudo-observation family.

#
# ### Decoupled Pseudo-Observation vs Non-Centered
#
# The relationship here is the most delicate, due to the restriction that
# ``\hat{\mathbf{S}}`` must be diagonal.
# This approximation achieves the optimal approximate posterior when the choice of
# pseudo observational data (``\hat{y}``, ``\hat{\mathbf{S}}``, and ``\mathbf{v}``) equal
# the original observational data.
# When the original observational data involves a non-Gaussian likelihood, this
# approximation family can still obtain the optimal approximate posterior provided that
# ``\mathbf{v}`` lines up with the inputs associated with the original data, ``\mathbf{x}``.
#
# To see this, consider the pseudo-observation approximation which makes use of the
# original observational data (generated at the top of this example):
decoupled_approx = PseudoObsSparseVariationalApproximation(f, z, Σ, x, y);
decoupled_posterior = posterior(decoupled_approx);

# We can get the optimal pseudo-point approximation using standard functionality:
optimal_approx_post = posterior(VFE(f(z, jitter)), f(x, Σ), y);

# The marginal statistics agree:
@assert mean(optimal_approx_post(x_pred)) ≈ mean(decoupled_posterior(x_pred))
@assert cov(optimal_approx_post(x_pred)) ≈ cov(decoupled_posterior(x_pred))

# The reason to think that this parametrisation will do something sensible is this property.
# Obviously when ``\mathbf{v} \neq \mathbf{x}`` the optimal approximate posterior cannot be
# recovered, however, when the hope is that there exists a small pseudo-dataset which gets
# close to the optimum.



# [^Titsias]: Titsias, M. K. [Variational learning of inducing variables in sparse Gaussian processes](https://proceedings.mlr.press/v5/titsias09a.html)
# [^Gorinova]: Gorinova, Maria and Moore, Dave and Hoffman, Matthew [Automatic Reparameterisation of Probabilistic Programs](http://proceedings.mlr.press/v119/gorinova20a)
# [^Paciorek]: [Paciorek, Christopher Joseph. Nonstationary Gaussian processes for regression and spatial modelling. Diss. Carnegie Mellon University, 2003.](https://www.stat.berkeley.edu/~paciorek/diss/paciorek-thesis.pdf)
