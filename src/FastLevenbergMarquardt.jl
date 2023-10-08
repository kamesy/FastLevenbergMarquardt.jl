module FastLevenbergMarquardt


using Base: require_one_based_indexing
using LinearAlgebra: BlasFloat, BlasInt, checksquare

using LinearAlgebra
using SparseArrays
using StaticArrays
using SuiteSparse


#####
##### Levenberg-Marquardt Workspace
#####

struct LMWorkspace{Tx, Tf, TJ, Tp, Tg, TD, Tw}
    x::Tx   # initial guess, solution
    f::Tf   # f = fun!(f, x, data)
    J::TJ   # J = jacobian!(J, x, data)
    p::Tp   # (J'J + λD'D) * p = g
    g::Tg   # g = J'f
    xk::Tx  # trial x
    fk::Tf  # trial f
    D::TD   # D'D diagonal scaling
    w::Tw   # workspace
end

function LMWorkspace(
    x::AbstractVector{<:AbstractFloat},
    m::Integer = length(x)
)
    n = length(x)
    f = similar(x, m)
    J = similar(x, (m, n))
    LMWorkspace(x, f, J)
end

function LMWorkspace(
    x::StaticVector{N, <:AbstractFloat},
    m::Integer = length(x)
) where {N}
    LMWorkspace(x, Val(m))
end

function LMWorkspace(
    x::StaticVector{N, <:AbstractFloat},
    ::Val{M},
) where {N, M}
    szf, szJ = M <= 12 ? (Size((M,)), Size((M, N))) : ((M,), (M, N))
    f = similar(x, szf)
    J = similar(x, szJ)
    LMWorkspace(x, f, J)
end

function LMWorkspace(
    x::AbstractVector{<:AbstractFloat},
    f::AbstractVector{<:AbstractFloat},
    J::AbstractMatrix{<:AbstractFloat},
)
    size(J, 1) == length(f) || throw(DimensionMismatch("size(J, 1) != length(f)"))
    size(J, 2) == length(x) || throw(DimensionMismatch("size(J, 2) != length(x)"))

    p  = similar(x)
    g  = similar(x)
    xk = similar(x)
    fk = similar(f)
    D  = similar(x)
    w  = similar(x)

    if x isa SVector
        x = MVector(x)
    end

    LMWorkspace(x, f, J, p, g, xk, fk, D, w)
end


include("solvers.jl")
include("lm.jl")


export LMWorkspace, CholeskySolver, QRSolver
export lmsolve, lmsolve!


end # module
