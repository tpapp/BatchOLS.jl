using BatchOLS
using Base.Test
import ForwardDiff: Dual
using Distributions

rand_Dual(M) = Dual(randn(M+1)...)

rand_Dual(N, M) = [rand_Dual(M) for _ in 1:N]

rand_yX(N, K) = randn(N), randn(N, K)

rand_yX_Dual(N, K, M) = rand_Dual(N, M), randn(N, K)

function alt_ML_βv(y, X)
    β = X \ y
    e = y-X*β
    β, dot(e,e)/size(X, 1)
end

alt_loglikelihood(y, X, β, v) = sum(logpdf.(Normal(0, √v), y.-(X*β)))

@testset "real ML" begin
    for _ in 1:100
        K = rand(2:5)
        N = rand(50:100) + K
        y, X = rand_yX(N, K)
        rhs = BatchOLS.RHS(X)
        β, v = BatchOLS.ML_βv(y, rhs)
        β′, v′ = alt_ML_βv(y, X)
        @test β ≈ β′
        @test v ≈ v
        @inferred BatchOLS.ML_βv(y, rhs)
        @test size(rhs) == size(X)
        @test size(rhs, 1) == size(X, 1)
        @test size(rhs, 2) == size(X, 2)
    end
end

@testset "dual ML" begin
    for _ in 1:100
        K = rand(2:5)
        N = rand(50:100) + K
        y, X = rand_yX_Dual(N, K, rand(1:3))
        rhs = BatchOLS.RHS(X)
        β, v = BatchOLS.ML_βv(y, rhs)
        β′, v′ = alt_ML_βv(y, X)
        @test β ≈ β′
        @test v ≈ v
        @inferred BatchOLS.ML_βv(y, rhs)
    end
end

@testset "real loglikelihood" begin
    for _ in 1:100
        K = rand(2:5)
        N = rand(50:100) + K
        y, X = rand_yX(N, K)
        rhs = BatchOLS.RHS(X)
        β = randn(K)
        v = (randn()+0.1)^2
        β′ = β + randn(length(β))
        v′ = v + abs(randn()/10)
        ℓ = BatchOLS.loglikelihood(y, rhs, β′, v′)
        ℓ′ = alt_loglikelihood(y, X, β′, v′)
        @test ℓ ≈ ℓ′
        @inferred BatchOLS.loglikelihood(y, rhs, β′, v′)
    end
end
