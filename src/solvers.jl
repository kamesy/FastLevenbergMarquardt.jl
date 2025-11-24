abstract type AbstractSolver end


#####
##### Cholesky
#####

struct CholeskySolver{S} <: AbstractSolver
    A::S # = J'J + λD'D
    JtJ::S

    function CholeskySolver(A::S, JtJ::S) where {S<:AbstractMatrix{<:AbstractFloat}}
        require_one_based_indexing(A, JtJ)
        checksquare(A)
        size(A) == size(JtJ) || throw(DimensionMismatch("size(A) != size(JtJ)"))
        new{S}(A, JtJ)
    end
end

CholeskySolver(n::Integer) = CholeskySolver(Float64, n)
CholeskySolver(::Type{T}, n::Integer) where {T<:AbstractFloat} =
    CholeskySolver(Matrix{T}(undef, (n, n)), Matrix{T}(undef, (n, n)))

CholeskySolver(A::AbstractMatrix{<:AbstractFloat}) =
    CholeskySolver(A, similar(A))


function init!(s::CholeskySolver, ::AbstractVector, L::LMWorkspace; newton::Bool)
    if newton
        copyto!(s.JtJ, L.J) # L.J is the Hessian
    else
        _mul!(s.JtJ, L.J', L.J) # L.J is the Jacobian
    end
    return s
end

update!(s::CholeskySolver, f::AbstractVector, L::LMWorkspace; newton::Bool) =
    init!(s, f, L; newton)

# solve (J'J + λD'D) * p = g = J'f
function solve!(s::CholeskySolver, λ::Real, ::AbstractVector, L::LMWorkspace)
    p, DtD, g, A, JtJ = L.p, L.D, L.g, s.A, s.JtJ
    if length(p) <= 12
        return _cholsolve!(p, JtJ, λ, DtD, g, A)
    end
    return cholsolve!(p, JtJ, λ, DtD, g, A)
end


function cholsolve!(
    p::AbstractVector,
    JtJ::AbstractMatrix,
    λ::Real,
    DtD::AbstractVector,
    g::AbstractVector,
    A::AbstractMatrix
)
    A = axpyA!(A, λ, DtD, JtJ)
    F, info = _cholesky!(A)
    if info == 0
        p = _ldiv!(p, F, g)
    end
    return p, info
end

_cholsolve!(
    p::StaticVector{N},
    JtJ::AbstractMatrix,
    λ::Real,
    DtD::StaticVector{N},
    g::StaticVector{N},
    ::StaticMatrix
) where {N} = __cholsolve!(Size((N, N)), Size((N,)), SVector(p), JtJ, λ, DtD, g)

_cholsolve!(
    p::StridedVector,
    JtJ::StaticMatrix{N, N},
    λ::Real,
    DtD::StridedVector,
    g::StridedVector,
    ::AbstractMatrix
) where {N} = __cholsolve!(Size((N, N)), Size((N,)), p, JtJ, λ, DtD, g)


function _cholsolve!(
    p::AbstractVector,
    JtJ::AbstractMatrix,
    λ::Real,
    DtD::AbstractVector,
    g::AbstractVector,
    A::AbstractMatrix
)
    N = length(p)

    # there has to be a better way of doing this
    if N == 1
        return __cholsolve!(Size((1, 1)), Size((1,)), p, JtJ, λ, DtD, g)
    elseif N == 2
        return __cholsolve!(Size((2, 2)), Size((2,)), p, JtJ, λ, DtD, g)
    elseif N == 3
        return __cholsolve!(Size((3, 3)), Size((3,)), p, JtJ, λ, DtD, g)
    elseif N == 4
        return __cholsolve!(Size((4, 4)), Size((4,)), p, JtJ, λ, DtD, g)
    elseif N == 5
        return __cholsolve!(Size((5, 5)), Size((5,)), p, JtJ, λ, DtD, g)
    elseif N == 6
        return __cholsolve!(Size((6, 6)), Size((6,)), p, JtJ, λ, DtD, g)
    elseif N == 7
        return __cholsolve!(Size((7, 7)), Size((7,)), p, JtJ, λ, DtD, g)
    elseif N == 8
        return __cholsolve!(Size((8, 8)), Size((8,)), p, JtJ, λ, DtD, g)
    elseif N == 9
        return __cholsolve!(Size((9, 9)), Size((9,)), p, JtJ, λ, DtD, g)
    elseif N == 10
        return __cholsolve!(Size((10, 10)), Size((10,)), p, JtJ, λ, DtD, g)
    elseif N == 11
        return __cholsolve!(Size((11, 11)), Size((11,)), p, JtJ, λ, DtD, g)
    elseif N == 12
        return __cholsolve!(Size((12, 12)), Size((12,)), p, JtJ, λ, DtD, g)
    else
        return cholsolve!(p, JtJ, λ, DtD, g, A)
    end
end


# solver (J'J + λD'D)*p = g
@generated function __cholsolve!(
    ::Size{sj},
    ::Size{sg},
    p::AbstractVector,
    JtJ::AbstractMatrix,
    λ::Real,
    DtD::AbstractVector,
    g::AbstractVector,
) where {sj, sg}
    N = sg[1]
    @assert sj[1] == N
    @assert sj[2] == N

    q = Expr(:block)

    # load arrays
    for n in 1:N
        xn = Symbol(:x, n)
        push!(q.args, :(@inbounds $xn = g[$n]))
    end

    # A = JtJ + λDtD
    for n in 1:N
        Ann = Symbol(:A, n, n)
        push!(q.args, :(@inbounds $Ann = muladd(λ, DtD[$n], JtJ[$n, $n])))
        push!(q.args, :($Ann >= zero($Ann) || return (p, $n)))
        for m in n+1:N
            Amn = Symbol(:A, m, n)
            push!(q.args, :(@inbounds $Amn = JtJ[$m, $n]))
        end
    end

    # "cholesky" with sqrt propagated through
    for n in 1:N
        for k in 1:n-1, m in n:N
            Akk = Symbol(:A, k, k)
            Amn = Symbol(:A, m, n)
            Amk = Symbol(:A, m, k)
            Ank = Symbol(:A, n, k)
            push!(q.args, :($Amn = muladd(-$Ank, $Amk/$Akk, $Amn)))
        end
    end

    # forward substitution
    for j in 1:N
        xj  = Symbol(:x, j)
        Ajj = Symbol(:A, j, j)
        for i in j+1:N
            xi  = Symbol(:x, i)
            Aij = Symbol(:A, i, j)
            push!(q.args, :($xi = muladd(-$xj, $Aij/$Ajj, $xi)))
        end
    end

    # back substitution
    for j in N:-1:1
        xj  = Symbol(:x, j)
        Ajj = Symbol(:A, j, j)
        push!(q.args, :($xj /= $Ajj))
        for i in j-1:-1:1
            xi  = Symbol(:x, i)
            Aji = Symbol(:A, j, i)
            push!(q.args, :($xi = muladd(-$xj, $Aji, $xi)))
        end
    end

    # return
    if p <: StaticVector
        ret = Expr(:tuple)
        for i in 1:N
            push!(ret.args, Symbol(:x, i))
        end
        if p <: MVector
            push!(q.args, :(p .= similar_type(p)($ret), 0))
        else
           push!(q.args, :(similar_type(p)($ret), 0))
        end
    else
        for i in 1:N
            xi = Symbol(:x, i)
            push!(q.args, :(@inbounds p[$i] = $xi))
        end
        push!(q.args, :(p, 0))
    end

    return Expr(:block, Expr(:meta, :inline), q)
end


@inline function _cholesky!(A)
    C = cholesky!(A, Val(false), check=false)
    return C, C.info
end

@inline function _cholesky!(A::StridedMatrix{<:BlasFloat})
    A, info = LAPACK.potrf!('U', A)
    return Cholesky(A, 'U', info), info
end

@inline function _cholesky!(A::StaticMatrix)
    C = StaticArrays._cholesky(Size(A), SMatrix(A), false)
    return C, C.info
end

@inline function _cholesky!(A::SparseMatrixCSC)
    C = cholesky(A, check=false)
    return C, Int(!issuccess(C))
end


@inline function _ldiv!(_, A::Cholesky{T, <:SMatrix{N, N, T}}, b) where{N, T}
    A \ SVector{N}(b)
end

@inline function _ldiv!(x, A::Cholesky{T, <:MMatrix{N, N, T}}, b) where{N, T}
    x .= A \ SVector{N}(b)
    return x
end

@inline function _ldiv!(x, A::SuiteSparse.CHOLMOD.Factor, b)
    x .= A \ b
    return x
end


@inline function axpyA!(A, λ, DtD, JtJ)
    A .= JtJ .+ λ .* Diagonal(DtD)
    return A
end

@inline function axpyA!(
    A::StridedMatrix,
    λ::Real,
    DtD::StridedVector,
    JtJ::AbstractMatrix,
)
    A = copyto!(A, JtJ)
    @simd ivdep for i in axes(A, 1)
        @inbounds A[i,i] += λ * DtD[i]
    end
    return A
end


#####
##### QR
#####

struct QRSolver{C, P, W} <: AbstractSolver
    tau::C
    jpvt::P
    work::W

    function QRSolver(tau::C, jpvt::P, work::W) where {
        C<:AbstractVector{T},
        P<:AbstractVector{BlasInt},
        W<:AbstractVector{T}
    } where {T<:BlasFloat}
        require_one_based_indexing(tau, jpvt, work)
        length(work) >= 2*length(tau) || throw(DimensionMismatch("length(work) < 2*N"))
        new{C, P, W}(tau, jpvt, work)
    end
end

QRSolver(n::Integer) = QRSolver(Float64, n)
QRSolver(::Type{T}, n::Integer) where {T<:BlasFloat} =
    QRSolver(Vector{T}(undef, n), Vector{BlasInt}(undef, n), Vector{T}(undef, 2*n))

QRSolver(tau::AbstractVector{T}) where {T<:BlasFloat} =
    QRSolver(tau, similar(tau, BlasInt), Vector{T}(undef, 2*length(tau)))

function QRSolver(jpvt::AbstractVector{BlasInt})
    T = float(eltype(jpvt))
    QRSolver(similar(jpvt, T), jpvt, Vector{T}(undef, 2*length(jpvt)))
end


function init!(
    s::QRSolver{<:AbstractArray{<:BlasFloat}},
    f::AbstractVector{<:BlasFloat},
    L::LMWorkspace,
    ::Val{lquery} = Val(true);
    newton::Bool = false
) where {lquery}
    newton && throw(ArgumentError("QRSolver does not support newton=true"))
    DtD, J, tau, jpvt, work = L.D, L.J, s.tau, s.jpvt, s.work
    jpvt = fill!(jpvt, 0)
    # QR decomp J
    J, tau, jpvt = _geqp3!(J, jpvt, tau, work, lquery)
    # precompute Q'f
    f = _ormqr!('L', 'T', J, tau, f, work, lquery)
    # precompute
    tau .= sqrt.(DtD)
    return s
end

update!(s::QRSolver, f::AbstractVector, L::LMWorkspace; newton::Bool = false) =
    init!(s, f, L, Val(false); newton)

# solve J*p = f, √λD*p = 0
function solve!(s::QRSolver, λ::Real, f::AbstractVector, L::LMWorkspace)
    p, J, D, jpvt, work = L.p, L.J, s.tau, s.jpvt, s.work
    return qrsolve!(p, J, f, λ, D, jpvt, work)
end


# translated from minpack
#   solve J*p = f, √λD*p = 0
# QR factorization of J precomputed, stored in J
# Q'f precomputed, stored in f
# computes least squares solution for singular system
function qrsolve!(
    p::AbstractVector{<:AbstractFloat},
    J::AbstractMatrix{<:AbstractFloat},
    f::AbstractVector{<:AbstractFloat},
    λ::Real,
    D::AbstractVector{<:AbstractFloat},
    jpvt::AbstractVector{BlasInt},
    w::AbstractVector{<:AbstractFloat},
)
    n = length(p)
    wf = view(w, 1:n)
    λD = view(w, n+1:2*n)

    zeroT = zero(zero(eltype(J))*zero(eltype(wf)))
    sqrtλ = sqrt(λ)

    # copy J and Q'f to preserve input and initialize wf.
    # in particular, save the diagonal elements of J in p.
    @inbounds for j in 1:n
        wf[j] = f[j]
        p[j]  = J[j,j]
        for i in j+1:n
            J[i,j] = J[j,i]
        end
    end

    # eliminate the diagonal matrix λD using a givens rotation.
    @inbounds for j in 1:n
        # prepare the row of λD to be eliminated, locating the
        # diagonal element using jpvt from the qr factorization.
        λD[j] = sqrtλ * D[jpvt[j]]

        for k in j+1:n
            λD[k] = 0
        end

        # the transformations to eliminate the row of λD
        # modify only a single element of Q'f
        # beyond the first n, which is initially zero.
        fpj = zeroT

        for k in j:n
            if λD[k] == 0
                continue
            end

            # determine a givens rotation which eliminates the
            # appropriate element in the current row of λD.
            rr = hypot(J[k,k], λD[k])
            cs = J[k,k] / rr
            sn = λD[k]  / rr

            # compute the modified diagonal element of J and
            # the modified element of [Q'f; 0].
            J[k,k] = rr

            wfk   =  wf[k]
            wf[k] =  cs*wfk + sn*fpj
            fpj   = -sn*wfk + cs*fpj

            # accumulate the tranformation in the row of λD.
            for i in k+1:n
                Jik, λDi = J[i,k], λD[i]
                J[i,k] =  cs*Jik + sn*λDi
                λD[i]  = -sn*Jik + cs*λDi
            end
        end

        # store the diagonal element of λD and restore
        # the corresponding diagonal element of J.
        λD[j]  = J[j,j]
        J[j,j] = p[j]
    end

    # solve the triangular system for z. if the system is
    # singular, then obtain a least squares solution.
    nsing = n
    @inbounds for i in 1:n
        if λD[i] == 0
            for j in i:n
                wf[j] = 0
                p[jpvt[j]] = 0
            end
            nsing = i-1
            break
        end
    end

    @inbounds for j in nsing:-1:1
        fj = wf[j]
        for i in j+1:nsing
            fj = muladd(-wf[i], J[i,j], fj)
        end
        wf[j] = fj / λD[j]
    end

    # permute the components of z back to components of x.
    @inbounds for j in 1:nsing
        p[jpvt[j]] = wf[j]
    end

    return p, 0
end


# TODO: generated qr
# The generated solve doesn't help as the LAPACK calls are dominating
#
# translated from minpack
#   solve J*p = f, √λD*p = 0
# QR factorization of J precomputed, stored in J
# Q'f precomputed, stored in f
# Note: no least squares solution for singular system
@generated function __qrsolve!(
    ::Size{sj},
    ::Size{sp},
    p::AbstractVector{<:AbstractFloat},
    J::AbstractMatrix{<:AbstractFloat},
    f::AbstractVector{<:AbstractFloat},
    λ::Real,
    D::AbstractVector{<:AbstractFloat},
    jpvt::AbstractVector{BlasInt},
    ijpvt::AbstractVector{BlasInt},
) where {sj, sp}
    N = sp[1]
    @assert sj[2] == N

    q = Expr(:block)
    push!(q.args, :(zeroT = zero(zero(eltype(J))*zero(eltype(f)))))

    # precompute
    push!(q.args, :(sqrtλ = sqrt(λ)))

    # load jpvt
    for i in 1:N
        jpvti = Symbol(:jpvt, i)
        push!(q.args, :(@inbounds $jpvti = jpvt[$i]))
    end

    # load Q'f
    for i in 1:N
        fi = Symbol(:f, i)
        push!(q.args, :(@inbounds $fi = f[$i]))
    end

    # load J
    for j in 1:N
        for i in j:N
            Jij = Symbol(:J, i, j)
            push!(q.args, :(@inbounds $Jij = J[$j, $i]))
        end
    end

    # singular check
    push!(q.args, :(ising = false))

    # eliminate the diagonal matrix λD using a givens rotation.
    for j in 1:N
        # prepare the row of λD to be eliminated, locating the
        # diagonal element using jpvt from the qr factorization.
        λDj = Symbol(:λD, j)
        jpvtj = Symbol(:jpvt, j)
        push!(q.args, :(@inbounds $λDj = sqrtλ * D[$jpvtj]))
        for k in j+1:N
            λDk = Symbol(:λD, k)
            push!(q.args, :($λDk = zeroT))
        end

        # the transformations to eliminate the row of λD
        # modify only a single element of Q'f
        # beyond the first n, which is initially zero.
        push!(q.args, :(fpj = zeroT))

        for k in j:N
            fk = Symbol(:f, k)
            Jkk = Symbol(:J, k, k)
            λDk = Symbol(:λD, k)
            ck, sk, rk = Symbol(:c, k), Symbol(:s, k), Symbol(:r, k)

            ex = Expr(:block)
            # determine a givens rotation which eliminates the
            # appropriate element in the current row of λD.
            push!(ex.args, quote
                $rk = hypot($Jkk, $λDk)
                $ck = $Jkk / $rk
                $sk = $λDk / $rk

                # compute the modified diagonal element of J and
                # the modified element of [Q'f; 0].
                $Jkk = $rk
                fk = $fk
                $fk =  $ck*fk + $sk*fpj
                fpj = -$sk*fk + $ck*fpj
            end)

            # accumulate the tranformation in the row of λD.
            for i in k+1:N
                Jik = Symbol(:J, i, k)
                λDi = Symbol(:λD, i)
                push!(ex.args, quote
                    Jik =$Jik
                    $Jik =  $ck*Jik + $sk*$λDi
                    $λDi = -$sk*Jik + $ck*$λDi
                end)
            end

            Jjj = Symbol(:J, j, j)
            push!(ex.args, :(ising |= $Jjj == 0))

            push!(q.args, quote
                if $λDk != 0
                    $ex
                end
            end)
        end
    end

    # solve the triangular system for z
    for j in N:-1:1
        fj = Symbol(:f, j)
        Jjj = Symbol(:J, j, j)
        push!(q.args, :($fj /= $Jjj))
        for i in j-1:-1:1
            fi = Symbol(:f, i)
            Jji = Symbol(:J, j, i)
            push!(q.args, :($fi = muladd(-$fj, $Jji, $fi)))
        end
    end

    # permute the components of z back to components of x.
    if p <: StaticVector
        ret = Expr(:tuple)
        for i in 1:N
            push!(ret.args, Symbol(:f, i))
        end
        if p <: MVector
            push!(q.args, :(p .= similar_type(p)($ret)[SVector{$N}(ijpvt)], Int(ising)))
        else
            push!(q.args, :(similar_type(p)($ret)[SVector{$N}(ijpvt)], Int(ising)))
        end
    else
        for i in 1:N
            fi = Symbol(:f, i)
            jpvti = Symbol(:jpvt, i)
            push!(q.args, :(p[$jpvti] = $fi))
        end
        push!(q.args, :(p, Int(ising)))
    end

    return Expr(:block, Expr(:meta, :inline), q)
end


#####
##### LinearAlgebra.LAPACK with preallocated workspace
#####

# libblastrampoline
@static if VERSION < v"1.7"
    const libblastrampoline = Base.liblapack_name
elseif VERSION < v"1.9"
    const libblastrampoline = "libblastrampoline"
else
    using LinearAlgebra: libblastrampoline
end


for (geqp3, elty) in ((:dgeqp3_, :Float64), (:sgeqp3_, :Float32))
    @eval begin
        # SUBROUTINE DGEQP3( M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, LDA, LWORK, M, N
        # *     .. Array Arguments ..
        #       INTEGER            JPVT( * )
        #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
        function _geqp3!(
            A::AbstractMatrix{$elty},
            jpvt::AbstractVector{BlasInt},
            tau::AbstractVector{$elty},
            work::Vector{$elty} = Vector{$elty}(undef, 1),
            lquery::Bool = true
        )
            Base.require_one_based_indexing(A, jpvt, tau)
            LinearAlgebra.chkstride1(A, jpvt, tau)

            m, n = size(A)
            lda = stride(A, 2)

            info = Ref{BlasInt}()
            lwork = length(work)

            if lquery
                lwork = BlasInt(-1)

                ccall(
                    (LinearAlgebra.BLAS.@blasfunc($geqp3), libblastrampoline),
                    Cvoid,
                    (
                        Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt},
                        Ptr{BlasInt},
                        Ptr{$elty},
                        Ptr{$elty},
                        Ref{BlasInt}, Ptr{BlasInt}
                    ),
                    m, n, A, lda, jpvt, tau, work, lwork, info
                )
                LAPACK.chklapackerror(info[])

                lwork = BlasInt(real(work[1]))
                if length(work) < lwork
                    resize!(work, lwork)
                end
            end

            ccall(
                (LinearAlgebra.BLAS.@blasfunc($geqp3), libblastrampoline),
                Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt},
                    Ptr{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt}, Ptr{BlasInt}
                ),
                m, n, A, lda, jpvt, tau, work, lwork, info
            )
            LAPACK.chklapackerror(info[])

            return A, tau, jpvt
        end
    end
end


for (ormqr, elty) in ((:dormqr_, :Float64), (:sormqr_, :Float32))
    @eval begin
        #      SUBROUTINE DORMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
        #                         WORK, INFO )
        #      .. Scalar Arguments ..
        #      CHARACTER          SIDE, TRANS
        #      INTEGER            INFO, K, LDA, LDC, M, N
        #      .. Array Arguments ..
        #      DOUBLE PRECISION   A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
        function _ormqr!(
            side::AbstractChar,
            trans::AbstractChar,
            A::AbstractMatrix{$elty},
            tau::AbstractVector{$elty},
            C::AbstractVecOrMat{$elty},
            work::Vector{$elty} = Vector{$elty}(undef, 1),
            lquery::Bool = true
        )
            Base.require_one_based_indexing(A, tau, C)
            LinearAlgebra.chkstride1(A, C, tau)

            m, n = size(C, 1), size(C, 2)
            k = length(tau)
            lda = stride(A, 2)
            ldc = stride(C, 2)

            info  = Ref{BlasInt}()
            lwork = length(work)

            if lquery
                lwork = BlasInt(-1)

                ccall(
                    (LinearAlgebra.BLAS.@blasfunc($ormqr), libblastrampoline),
                    Cvoid,
                    (
                        Ref{UInt8}, Ref{UInt8},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty},
                        Ref{BlasInt}, Ptr{BlasInt}
                    ),
                    side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info
                )
                LAPACK.chklapackerror(info[])

                lwork = BlasInt(real(work[1]))
                if length(work) < lwork
                    resize!(work, lwork)
                end
            end

            ccall(
                (LinearAlgebra.BLAS.@blasfunc($ormqr), libblastrampoline),
                Cvoid,
                (
                    Ref{UInt8}, Ref{UInt8},
                    Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt}, Ptr{BlasInt}
                ),
                side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info
            )
            LAPACK.chklapackerror(info[])

            return C
        end
    end
end
