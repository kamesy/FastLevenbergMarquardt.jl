using FastLevenbergMarquardt

using Statistics: mean
using Aqua
using LinearAlgebra: dot, eigmin, norm
using Printf
using StaticArrays
using Test


@testset "FastLevenbergMarquardt.jl" begin
    Aqua.test_all(
        FastLevenbergMarquardt,
        deps_compat=(ignore=[:LinearAlgebra, :SparseArrays, :SuiteSparse],)
    )
end


@testset "NIST" begin
    include("nist.jl")

    blre(x, y) = clamp(minimum(-log10.(abs.(x .- y)./abs.(y))), 0, 11)

    @testset "Default Solver" begin
        @printf("\nNIST - Default Solver\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                res1 = lmsolve!(P.fun!, P.jac!, copy(b0), P.m, P.data)
                res2 = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data)
                @test res1[1:end-2] == res2[1:end-2]

                b, F, converged, iter, nfev, njev, LM, solver = res1
                res3 = lmsolve!(P.fun!, P.jac!, (LM.x .= b0; LM), P.data, solver=solver)
                @test res3[1:end-2] == res1[1:end-2]

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
            for (i, b0) in enumerate(P.b0)
                b0 = SVector{P.n}(b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, b0, P.m, P.data)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end

    @testset "Dense Cholesky" begin
        @printf("\nNIST - Dense Cholesky\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data, solver=:cholesky)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0

                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end

    @testset "Dense QR" begin
        @printf("\nNIST - Dense QR\n\n")
        @printf("%17s lre      rss    conv  iter  nfev  njev\n", "")
        resv = []

        problems = []
        for P in Problems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for (i, b0) in enumerate(P.b0)
                b, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, copy(b0), r, J, P.data, solver=:qr)

                lre = blre(b, P.b)
                @test F ≈ P.rss rtol = 1e-10 atol = 1e-20
                @test lre > 5
                @test converged > 0
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
                @printf("%12s %d   %4.1f  %.4e  %2d  %4d  %4d  %4d\n",
                    i == 1 ? P.name * " -" : "", i, lre, F, converged, iter, nfev, njev)
            end
        end

        @printf("\n")
        meanlrev = mean(r -> r[3], resv)
        minlrev  = minimum(r -> r[3], resv)
        @printf("mean log relative error = %4.1f, min lre = %4.1f\n", meanlrev, minlrev)
        @printf("\n")
    end
end


@testset "MGH" begin
    include("mgh.jl")

    flre(x, y) = maximum(-log10.(abs.(x .- y)./ifelse.(y .== 0, 1, abs.(y))))

    @testset "Default Solver" begin
        @printf("\nMGH - Default Solver\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                x0 = i.*P.x0

                res1 = lmsolve!(P.fun!, P.jac!, copy(x0), P.m, P.data)
                res2 = lmsolve!(P.fun!, P.jac!, copy(x0), r, J, P.data)
                @test res1[1:end-2] == res2[1:end-2]

                x, F, converged, iter, nfev, njev, LM, solver = res1
                res3 = lmsolve!(P.fun!, P.jac!, (LM.x .= x0; LM), P.data, solver=solver)
                @test res3[1:end-2] == res1[1:end-2]

                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, Float64, BigFloat)
            x0 = Array{T}(P.x0)
            x, F, converged, iter, nfev, njev, = lmsolve!(P.fun!, P.jac!, x0)
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)

            x0 = SVector{12, T}(P.x0)
            fun = (x, data) -> SVector{12}(P.fun!(zeros(T, 12), x, data))
            jac = (x, data) -> SMatrix{12, 12}(P.jac!(zeros(T, 12, 12), x, data))

            x, F, converged, iter, nfev, njev = lmsolve(fun, jac, x0)
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)
        end
    end

    @testset "Dense Cholesky" begin
        @printf("\nMGH - Dense Cholesky\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                res = lmsolve!(P.fun!, P.jac!, i.*P.x0, r, J, P.data, solver=:cholesky)
                x, F, converged, iter, nfev, njev, LM, solver = res
                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, BigFloat)
            x0 = Array{T}(P.x0)
            res = lmsolve!(P.fun!, P.jac!, x0, solver=:cholesky)
            x, F, converged, iter, nfev, njev, LM, solver = res
            @test F ≈ 0 atol=1e-30
            @test x ≈ ones(T, length(x0)) rtol=1e-16
            @test F isa T
            @test x isa typeof(x0)
        end
    end

    @testset "Dense QR" begin
        @printf("\nMGH - Dense QR\n\n")
        resv = []

        for P in TestProblems
            r = zeros(P.m)
            J = zeros(P.m, P.n)

            for i in (1, 10, 100)
                res = lmsolve!(P.fun!, P.jac!, i.*P.x0, r, J, P.data, solver=:qr)
                x, F, converged, iter, nfev, njev, LM, solver = res
                lre = flre(F, P.fx)
                push!(resv, (P.name, i, lre, F, converged, iter, nfev, njev))
            end
        end

        n = length(resv)
        nsv = count(r -> r[3] > 4, resv)
        @test nsv > 130
        @printf("%3d/%3d with ||f(x)||_2 to at least 4 digits\n", nsv, n)
        @printf("\n")

        # Test Float32 and BigFloat
        P = RosenbrockN(12)

        for T in (Float32, BigFloat)
            x0 = Array{T}(P.x0)
            if T === BigFloat
                @test_throws ArgumentError lmsolve!(P.fun!, P.jac!, x0, solver=:qr)
            else
                res = lmsolve!(P.fun!, P.jac!, x0, solver=:qr)
                x, F, converged, iter, nfev, njev, LM, solver = res
                @test F ≈ 0 atol=1e-30
                @test x ≈ ones(T, length(x0)) rtol=1e-16
                @test F isa T
                @test x isa typeof(x0)
            end
        end
    end
end


@testset "Newton" begin
    tol_x = 1e-8
    tol_F = 1e-24
    tol_g = 1e-8

    #### Convex quadratic (unique minimizer)

    # φ(x) = 1/2 x'Qx + b'x
    # ∇φ = Qx + b
    # ∇²φ = Q
    @testset "Convex Quadratic" begin
        Q = @SMatrix [4.0 1.0 0.0; 1.0 3.0 0.2; 0.0 0.2 2.0]
        b = @SVector [-1.0, 2.0, -0.5]

        φ(x, p=nothing) = 0.5 * (x' * (Q * x)) + dot(b, x)
        grad(x, p=nothing) = Q * x + b
        withgrad(x, p=nothing) = φ(x, p), grad(x, p)
        hess(x, p=nothing) = Q

        function grad!(f, x, p=nothing)
            f .= Q * x .+ b
            return f
        end
        function withgrad!(f, x, p=nothing)
            grad!(f, x, p)
            return φ(x, p), f
        end
        function hess!(J, x, p=nothing)
            J .= Q
            return J
        end
        function withhess!(J, x, p=nothing)
            hess!(J, x, p)
            return φ(x, p), grad(x, p), J
        end

        x0s = @SVector [2.0, -1.0, 0.5]
        x0v = collect(x0s)
        xstar = -collect(Q \ b)

        @testset "Cholesky vs QR" begin
            n = length(x0v)
            fbuf = zeros(n)
            Jbuf = zeros(n, n)

            # LM with QR
            xLM_qr, FLM_qr, infoLM_qr, itLM_qr, nfevLM_qr, njevLM_qr, LMLM_qr, solLM_qr =
                lmsolve!(grad!, hess!, copy(x0v), copy(fbuf), copy(Jbuf); newton=false, solver=:qr)

            # LM with Cholesky
            xLM_ch, FLM_ch, infoLM_ch, itLM_ch, nfevLM_ch, njevLM_ch, LMLM_ch, solLM_ch =
                lmsolve!(grad!, hess!, copy(x0v), copy(fbuf), copy(Jbuf); newton=false, solver=:cholesky)

            # Newton (Cholesky only)
            xN_ch, FN_ch, infoN_ch, itN_ch, nfevN_ch, njevN_ch, LMN_ch, solN_ch =
                lmsolve!(withgrad!, withhess!, copy(x0v), copy(fbuf), copy(Jbuf); newton=true, solver=:cholesky)

            @test norm(xLM_qr - xstar) ≤ tol_x
            @test norm(xLM_ch - xstar) ≤ tol_x
            @test norm(xN_ch - xstar) ≤ tol_x
            @test FLM_qr ≤ tol_F
            @test FLM_ch ≤ tol_F
            @test FN_ch ≤ tol_F

            # QR must be rejected for Newton
            @test_throws ArgumentError lmsolve!(
                withgrad!, withhess!, copy(x0v), copy(fbuf), copy(Jbuf);
                newton=true, solver=:qr,
            )
        end
    end

    #### Nonconvex example (multiple stationary points)

    # φ(x₁,x₂) = x₁^4 - x₁^2 + x₂^2
    # ∇φ = [4x₁^3 - 2x₁, 2x₂]
    # ∇²φ = diag(12x₁^2 - 2, 2)
    @testset "Nonconvex" begin
        φ(x, p=nothing) = x[1]^4 - x[1]^2 + x[2]^2
        grad(x, p=nothing) = @SVector [4x[1]^3 - 2x[1], 2x[2]]
        withgrad(x, p=nothing) = φ(x, p), grad(x, p)
        function hess(x, p=nothing)
            d11 = 12x[1]^2 - 2
            @SMatrix [d11 0.0; 0.0 2.0]
        end
        function grad!(f, x, p=nothing)
            f[1] = 4x[1]^3 - 2x[1]
            f[2] = 2x[2]
            return f
        end
        function withgrad!(f, x, p=nothing)
            grad!(f, x, p)
            return φ(x, p), f
        end
        function hess!(J, x, p=nothing)
            J[1, 1] = 12x[1]^2 - 2
            J[1, 2] = 0.0
            J[2, 1] = 0.0
            J[2, 2] = 2.0
            return J
        end
        function withhess!(J, x, p=nothing)
            hess!(J, x, p)
            return φ(x, p), grad(x, p), J
        end

        xs = @SVector [0.8, 0.05]
        xv = collect(xs)

        @testset "Stationary Point" begin
            n = length(xv)
            fbuf = zeros(n)
            Jbuf = zeros(n, n)

            xN, FN, infoN, itN, nfevN, njevN, LMN, solN =
                lmsolve!(withgrad!, withhess!, copy(xv), copy(fbuf), copy(Jbuf); newton=true, solver=:cholesky)
            xL, FL, infoL, itL, nfevL, njevL, LML, solL =
                lmsolve!(grad!, hess!, copy(xv), copy(fbuf), copy(Jbuf); newton=false, solver=:cholesky)

            @test norm(grad!(fbuf, xN), Inf) ≤ tol_g
            @test norm(grad!(fbuf, xL), Inf) ≤ tol_g
            @test φ(xN) ≤ φ(xv) + 1e-8
            @test φ(xL) ≤ φ(xv) + 1e-8
        end
    end

    #### Pathological case: LM can converge to a saddle, Newton escapes to a min

    # φ(x,y) = x^4 + y^4 - 2 x^2 + 2 y^2
    # ∇φ = (4x^3 - 4x, 4y^3 + 4y)
    # ∇²φ = diag(12x^2 - 4, 12y^2 + 4)
    @testset "Pathological" begin
        φ(x, p=nothing) = x[1]^4 + x[2]^4 - 2x[1]^2 + 2x[2]^2
        grad(x, p=nothing) = @SVector [4x[1]^3 - 4x[1], 4x[2]^3 + 4x[2]]
        withgrad(x, p=nothing) = φ(x, p), grad(x, p)
        hess(x, p=nothing) = @SMatrix [12x[1]^2-4 0.0; 0.0 12x[2]^2+4]

        function grad!(f, x, p=nothing)
            f[1] = 4x[1]^3 - 4x[1]
            f[2] = 4x[2]^3 + 4x[2]
            return f
        end
        function withgrad!(f, x, p=nothing)
            grad!(f, x, p)
            return φ(x, p), f
        end
        function hess!(J, x, p=nothing)
            J[1, 1] = 12x[1]^2 - 4
            J[1, 2] = 0.0
            J[2, 1] = 0.0
            J[2, 2] = 12x[2]^2 + 4
            return J
        end
        function withhess!(J, x, p=nothing)
            hess!(J, x, p)
            return φ(x, p), grad(x, p), J
        end

        # Helper predicates
        is_saddle(x) = (eigmin(Matrix(hess(x))) < -1) && (norm(grad(x), Inf) ≤ 1e-8)
        near_min(x) = min(norm(x .- SVector(1.0, 0.0)), norm(x .- SVector(-1.0, 0.0))) ≤ 1e-8

        seeds = [
            SVector(0.20, 0.05),
            SVector(0.15, 0.10),
            SVector(0.30, 0.05),
            SVector(-0.20, 0.05),
            SVector(-0.15, 0.10),
            SVector(-0.30, 0.05),
            SVector(0.12, 0.02),
            SVector(-0.12, 0.02),
        ]

        @testset "Saddle vs Min" begin
            for xs in seeds
                n = length(xs)
                xv = collect(xs)
                fbuf = zeros(n)
                Jbuf = zeros(n, n)

                xLM, FLM, infoLM, itLM, nfevLM, njevLM, LMLM, solLM =
                    lmsolve!(grad!, hess!, copy(xv), copy(fbuf), copy(Jbuf); newton=false, solver=:cholesky)
                xN, FN, infoN, itN, nfevN, njevN, LMN, solN =
                    lmsolve!(withgrad!, withhess!, copy(xv), copy(fbuf), copy(Jbuf); newton=true, solver=:cholesky)

                @test is_saddle(xLM)
                @test !near_min(xLM)
                @test !is_saddle(xN)
                @test near_min(xN)
            end
        end
    end
end
