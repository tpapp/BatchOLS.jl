# BatchOLS

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tpapp/BatchOLS.jl.svg?branch=master)](https://travis-ci.org/tpapp/BatchOLS.jl)
[![Coverage Status](https://coveralls.io/repos/tpapp/BatchOLS.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/BatchOLS.jl?branch=master)
[![codecov.io](http://codecov.io/github/tpapp/BatchOLS.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/BatchOLS.jl?branch=master)

Maximum likelihood estimation and likelihood calculations for regressions of the form
```
y = X⋅β + ϵ

ϵ ∼ Normal(0, v), IID
```
where `β` are the coefficients and `v` is the variance of the error term.

The key features are

1. type stability, also for `ForwardDiff.Dual` numbers,

2. fast calculations for regressions using the same `X`.

Example:
```julia
import BatchOLS # no exported symbols

N = 100
K = 3
X = randn(N, K)
rhs = BatchOLS.RHS(X)

## maximum likelihood estimation
for _ in 1:100
    y = randn(N)
    β, v = BatchOLS.ML_βv(y, rhs)
end

## loglikelihood calculation
β′ = randn(K)
y = randn(N)
ℓ = BatchOLS.loglikelihood(y, rhs, β′, 1.0)
```

The code is fairly trivial, I put it in a package to allow rigorous automated testing after optimizations, especially for type inference.
