"""
Likelihood-based calculations for ordinary least squares for the same right hand side (covariates) repeatedly.
"""
module BatchOLS

using ArgCheck

"""
    RHS(X, [Xfact])

Wrapper for covariates of an ordinary least squares (OLS) regression that precalculates the necessary factorizations. The latter can be supplied, but is not checked for consistency.
"""
struct RHS{TX, TXfact}
    "Regressors (covariates)."
    X::TX
    "Factorization of X that allows solving the least squares problem."
    Xfact::TXfact
end

function RHS(X::AbstractMatrix)
    N, K = size(X)
    @argcheck N ≥ K "Need more rows than columns."
    RHS(X, qrfact(X, Val{true}))
end

Base.size(rhs::RHS, dim...) = size(rhs.X, dim...)

Base.:*(rhs::RHS, β) = rhs.X * β

"""
    β, v = ML_βv(y, rhs)

Maximum likelihood estimate of

``y = X⋅β + ϵ``

where ``ϵ ∼ Normal(0, v)``, IID, and `v` is the variance. Use `RHS` to pre-calculate `rhs`.
"""
function ML_βv(y, rhs::RHS)
    β = rhs.Xfact \ y
    e = y-rhs.X*β
    β, mean(abs2, e)
end

"""
    loglikelihood(y, rhs, β, v)

Log likelihood of observations under given parameters `β`, `v` (not necessarily ML estimates). See `ML_βv`.
"""
loglikelihood(y, rhs::RHS, β, v) =
    -(size(rhs, 1)/2*(log(2*π)+log(v)) + sum(abs2, y-rhs.X*β)/(2*v))

end # module
