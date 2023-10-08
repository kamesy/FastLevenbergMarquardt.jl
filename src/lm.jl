"""
    lmsolve(
        fun,
        jac,
        x0::StaticVector{N, <:AbstractFloat},
        data = nothing,
        lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
        kwargs...
    ) -> x, F, info, iter, nfev, njev

    lmsolve!(
        fun!,
        jac!,
        x0::AbstractVector{<:AbstractFloat},
        m::Integer = length(x0),
        data = nothing,
        lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
        kwargs...,
    ) -> x, F, info, iter, nfev, njev, LM, solver

    lmsolve!(
        fun!,
        jac!,
        x0::AbstractVector{<:AbstractFloat},
        f::AbstractVector{<:AbstractFloat},
        J::AbstractMatrix{<:AbstractFloat},
        data = nothing,
        lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
        kwargs...,
    ) -> x, F, info, iter, nfev, njev, LM, solver

    lmsolve!(
        fun!,
        jac!,
        LM::LMWorkspace,
        data = nothing,
        lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
        kwargs...
    ) -> x, F, info, iter, nfev, njev, LM, solver

Minimize `F(x) = ||f(x)||^2` using the Levenberg-Marquardt algorithm.

### Arguments
- `fun/fun!`: function to be minimized `||f||^2`, `f = fun(x, data)`,
    `f = fun!(f, x, data)`
- `jac/jac!`: jacobian of `f`, `J = jac(x, data)`, `J = jac!(J, x, data)`
- `x0::AbstractVector{<:AbstractFloat}`: initial guess
- `m::Integer = length(x0)`: number of function values
- `data = nothing`: data passed to `fun/fun!` and `jac/jac!`
- `f::AbstractVector{<:AbstractFloat}`: preallocated function vector
- `J::AbstractMatrix{<:AbstractFloat}`: preallocated Jacobian matrix
- `LM::LMWorkspace`: preallocated workspace
- `lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing`: lower bounds
    for `x`. Vectors must have same length as `x`
- `ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing`: upper bounds
    for `x`. Vectors must have same length as `x`

### Keywords
- `solver::Union{Nothing, Symbol} = nothing`: linear solver for `lmsolve!`
        (`lmsolve` always uses Cholesky factorization)
    - `nothing`: QR for dense Jacobian, Cholesky for sparse Jacobian
    - `:cholesky`: Cholesky factorization
    - `:qr`: QR factorization
- `ftol::Real = eps(eltype(x))`: relative tolerance for function:
    both actual and predicted reductions are less than `ftol`
- `xtol::Real = 1e-10`: relative tolerance for change in `x`:
    `all(abs(x - xk) < xtol * (xtol + abs(x)))`
- `gtol::Real = eps(eltype(x))`: tolerance for gradient:
    `norm(g, Inf) < gtol`
- `maxit::Integer = 1000`: maximum number of iterations
- `factor::Real = 1e-6`: initial factor for damping
- `factoraccept::Real = 13`: factor for decreasing damping on good step
- `factorreject::Real = 3`: factor for increasing damping on bad step
- `factorupdate::Symbol = :marquardt`: factor update method
    `∈ (:marquardt, :nielsen)`
- `minscale::Real = 1e-12`: diagonal scaling lower bound
- `maxscale::Real = 1e16`: diagonal scaling upper bound
- `minfactor::Real = 1e-28`: damping factor lower bound
- `maxfactor::Real = 1e32`: damping factor upper bound

### Returns
- `x/LM.x`: solution
- `F`: final objective
- `info::Int`: convergence status
    - `1`: both actual and predicted reductions are less than `ftol`
    - `2`: relative difference between two consecutive iterates is less than `xtol`
    - `3`: inf norm of the gradient is less than `gtol`
    - `-1`: `maxit` reached
- `iter::Int`: number of iterations
- `nfev::Int`: number of function evaluations
- `njev::Int`: number of Jacobian evaluations
- `LM::LMWorkspace`: workspace
- `solver::AbstractSolver`: solver

### Notes
In the returned `LMWorkspace`, only `LM.x` and `LM.f` are guaranteed to be
updated. That is, `LM.J` might not be the Jacobian at the returned `x`.
"""
lmsolve, lmsolve!


function lmsolve(
    fun::FUN,
    jac::JAC,
    x0::StaticVector{N, <:AbstractFloat},
    data::P = nothing,
    lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
    kwargs...
) where {FUN, JAC, N, P}
    if lb isa AbstractVector
        length(lb) == N || throw(DimensionMismatch("length(lb) != length(x)"))
    end
    if ub isa AbstractVector
        length(ub) == N || throw(DimensionMismatch("length(ub) != length(x)"))
    end
    if lb !== nothing && ub !== nothing
        all(lb .<= ub) || throw(ArgumentError("lb > ub"))
        !all(lb .== ub) || throw(ArgumentError("lb == ub"))
    end

    _lmsolve(fun, jac, x0, data, lb, ub; kwargs...)
end

function lmsolve!(
    fun!::FUN,
    jac!::JAC,
    x0::AbstractVector{<:AbstractFloat},
    m::Integer = length(x0),
    data::P = nothing,
    lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
    kwargs...,
) where {FUN, JAC, P}
    LM = LMWorkspace(x0, m)
    lmsolve!(fun!, jac!, LM, data, lb, ub; kwargs...)
end

function lmsolve!(
    fun!::FUN,
    jac!::JAC,
    x0::AbstractVector{<:AbstractFloat},
    f::AbstractVector{<:AbstractFloat},
    J::AbstractMatrix{<:AbstractFloat},
    data::P = nothing,
    lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
    kwargs...,
) where {FUN, JAC, P}
    LM = LMWorkspace(x0, f, J)
    lmsolve!(fun!, jac!, LM, data, lb, ub; kwargs...)
end

function lmsolve!(
    fun!::FUN,
    jac!::JAC,
    LM::LMWorkspace,
    data::P = nothing,
    lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
    solver::Union{Nothing, Symbol, AbstractSolver} = nothing,
    kwargs...
) where {FUN, JAC, P}
    x, J = LM.x, LM.J
    n = length(x)
    @assert n == size(J, 2)

    if solver === nothing
        if J isa SparseMatrixCSC || !(eltype(J) <: BlasFloat)
            solver = :cholesky
        else
            solver = :qr
        end
    end

    if solver isa Symbol
        if solver === :cholesky
            solver = CholeskySolver(similar(x, eltype(J), x isa StaticArray ? Size((n, n)) : (n, n)))
        elseif solver === :qr
            J isa SparseMatrixCSC && throw(ArgumentError(":qr for sparse Jacobian not implemented"))
            eltype(J) <: BlasFloat || throw(ArgumentError(":qr requires Float32 or Float64 arrays"))
            solver = QRSolver(similar(J, J isa StaticArray ? Size((n,)) : (n,)))
        else
            throw(ArgumentError("solver must be one of :cholesky, :qr"))
        end
    end

    if solver isa CholeskySolver
        size(solver.JtJ, 1) == n || throw(DimensionMismatch("size(JtJ, 1) != size(J, 2)"))
        size(solver.JtJ, 2) == n || throw(DimensionMismatch("size(JtJ, 2) != size(J, 2)"))
    elseif solver isa QRSolver
        length(solver.tau)  == n || throw(DimensionMismatch("length(tau) != size(J, 2)"))
        length(solver.jpvt) == n || throw(DimensionMismatch("length(jpvt) != size(J, 2)"))
        size(LM.J, 2) <= size(LM.J, 1) || throw(ArgumentError("QRSolver not implemented for underdetermined problems"))
    end

    if lb isa AbstractVector
        length(lb) == n || throw(DimensionMismatch("length(lb) != length(x)"))
    end
    if ub isa AbstractVector
        length(ub) == n || throw(DimensionMismatch("length(ub) != length(x)"))
    end
    if lb !== nothing && ub !== nothing
        all(lb .< ub) || throw(ArgumentError("lb >= ub"))
        !all(lb .== ub) || throw(ArgumentError("lb == ub"))
    end

    _lmsolve!(fun!, jac!, LM, data, lb, ub, solver; kwargs...)
end


function _lmsolve!(
    fun!::FUN,
    jacobian!::JAC,
    LM::LMWorkspace{Tx},
    data::P,
    lb::Union{Nothing, Real, AbstractVector{<:Real}},
    ub::Union{Nothing, Real, AbstractVector{<:Real}},
    solver::AbstractSolver;
    ftol::Real = eps(eltype(Tx)),
    xtol::Real = 1e-10,
    gtol::Real = eps(eltype(Tx)),
    maxit::Integer = 1000,
    factor::Real = 1e-6,
    factoraccept::Real = 13,
    factorreject::Real = 3,
    factorupdate::Symbol = :marquardt,
    minscale::Real = 1e-12,
    maxscale::Real = 1e16,
    minfactor::Real = 1e-28,
    maxfactor::Real = 1e32,
) where {FUN, JAC, Tx, P}
    x, p, g, f, xk, fk = LM.x, LM.p, LM.g, LM.f, LM.xk, LM.fk
    J, DtD, w = LM.J, LM.D, LM.w

    T = eltype(Tx)
    zeroT = zero(T)

    # tolerances
    ϵc = convert(T, ftol)
    ϵx = convert(T, xtol)
    ϵg = convert(T, gtol)

    # scaling parameter, update factors, limits
    λ = convert(T, factor)
    λrej = convert(T, factorreject)
    λacc = convert(T, 1 / factoraccept)

    λlo = convert(T, minfactor)
    λhi = convert(T, maxfactor)

    Dlo = convert(T, minscale)
    Dhi = convert(T, maxscale)

    # factor for Nielsen update
    ν = λrej

    # function evals
    nfev = 0
    njev = 0

    # project initial guess
    if lb !== nothing
        @. x = max(x, lb)
    end

    if ub !== nothing
        @. x = min(x, ub)
    end

    # compute f(x)
    f = fun!(f, x, data)
    nfev += 1

    # initial objective
    F = sum(abs2, f)

    # early exit
    if F < ϵc
        converged = 1
        return x, F, converged, 0, nfev, njev, LM, solver
    end

    # compute jacobian J
    J = jacobian!(J, x, data)
    njev += 1

    # compute g = J'f
    g = _mul!(g, J', f)

    # early exit
    if norm(g, Inf) < ϵg
        converged = 3
        return x, F, converged, 0, nfev, njev, LM, solver
    end

    # init diagonal scaling
    DtD = vec(sum!(abs2, DtD', J))
    DtD = clamp!(DtD, Dlo, Inf)

    # init solver
    solver = init!(solver, f, LM)

    # main loop
    iter = 0
    converged = -1

    while iter < maxit
        iter += 1
        accepted = false

        # solve (J'J + λD'D) * p = J'f
        p, info = solve!(solver, λ, f, LM)

        if info == 0
            # project
            if lb !== nothing
                @. p = min(x-lb, p)
                all(iszero, p) && @goto reject
            end

            if ub !== nothing
                @. p = max(x-ub, p)
                all(iszero, p) && @goto reject
            end

            # evaluate trial step
            xk .= x .- p

            fk = fun!(fk, xk, data)
            Fk = sum(abs2, fk)
            nfev += 1

            ρ = zeroT
            # ρ = actual reduction / predicted reduction
            #   = (||f(x)|| - ||f(xk)||) / (||f(x)|| - ||f(x) + J(x)p||)
            #   = (1 - Fk/F) / ( ||Jp||/F + 2λ||Dp||/F)
            #   = (1 - Fk/F) / (λ||Dp||/F - <p,g>/F)
            #   = (F - Fk)   / <p, λD'Dp - g>
            if Fk < F
                ac = F - Fk
                pr = dot(p, (@. w = λ*DtD*p + g))
                ρ = ac / pr
                accepted = true
            end

            # check step size
            if all(>(0), (@. w = abs(p) < ϵx * (ϵx + abs(x))))
                if accepted
                    x, f, F = xk, fk, Fk
                end
                converged = 2
                break
            end
        end

        if accepted
            # update x = x + p
            x, xk = xk, x

            # update f = f(x + p)
            f, fk = fk, f

            # update objective
            F, Fk = Fk, F

            # check objective
            if abs(ac) < ϵc*Fk && pr < ϵc*Fk && ρ < 2
                converged = 1
                break
            end

            # jacobian J = J(x + p)
            J = jacobian!(J, x, data)
            njev += 1

            # g = J'f
            g = _mul!(g, J', f)

            # check gradient
            if norm(g, Inf) < ϵg
                converged = 3
                break
            end

            # update scaling
            w = vec(sum!(abs2, w', J))
            DtD .= clamp.(w, DtD, Dhi)

            # update solver
            solver = update!(solver, f, LM)

            # update λ
            if factorupdate === :nielsen
                ν = 2*ρ - 1
                λ *= max(λacc, 1 - (λrej-1)*ν*ν*ν)
                ν = λrej

            else # :marquardt
                if ρ > 0.75
                    λ *= λacc
                elseif ρ < 1e-3
                    λ *= λrej
                end
            end

        else
            @label reject
            # update λ
            if factorupdate === :nielsen
                λ *= ν
                ν *= 2
            else # :marquardt
                λ *= λrej
            end
        end

        λ = clamp(λ, λlo, λhi)
    end

    if x !== LM.x
        x = copyto!(LM.x, x)
        f = copyto!(LM.f, f)
    end

    return x, F, converged, iter, nfev, njev, LM, solver
end


function _lmsolve(
    fun::FUN,
    jacobian::JAC,
    x0::StaticVector{N, T},
    data::P,
    lb::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    ub::Union{Nothing, Real, AbstractVector{<:Real}} = nothing;
    ftol::Real = eps(T),
    xtol::Real = 1e-10,
    gtol::Real = eps(T),
    maxit::Integer = 1000,
    factor::Real = 1e-6,
    factoraccept::Real = 13,
    factorreject::Real = 3,
    factorupdate::Symbol = :marquardt,
    minscale::Real = 1e-12,
    maxscale::Real = 1e16,
    minfactor::Real = 1e-28,
    maxfactor::Real = 1e32,
) where {FUN, JAC, N, T, P}
    zeroT = zero(T)

    # tolerances
    ϵc = convert(T, ftol)
    ϵx = convert(T, xtol)
    ϵg = convert(T, gtol)

    # scaling parameter, update factors, limits
    λ = convert(T, factor)
    λrej = convert(T, factorreject)
    λacc = convert(T, 1 / factoraccept)

    λlo = convert(T, minfactor)
    λhi = convert(T, maxfactor)

    Dlo = convert(T, minscale)
    Dhi = convert(T, maxscale)

    # factor for Nielsen update
    ν = λrej

    # function evals
    nfev = 0
    njev = 0

    # project initial guess
    x = SVector(x0)

    if lb !== nothing
        x = max.(x, lb)
    end

    if ub !== nothing
        x = min.(x, ub)
    end

    # compute f(x)
    f = fun(x, data)
    nfev += 1

    # initial objective
    F = sum(abs2, f)

    # early exit
    if F < ϵc
        converged = 1
        return x, F, converged, 0, nfev, njev
    end

    # compute jacobian J
    J = jacobian(x, data)
    njev += 1

    # compute g = J'f
    g = J'*f

    # early exit
    if norm(g, Inf) < ϵg
        converged = 3
        return x, F, converged, 0, nfev, njev
    end

    # init diagonal scaling
    JtJ = J'*J
    DtD = diag(JtJ)
    DtD = clamp.(DtD, Dlo, Inf)

    # main loop
    iter = 0
    converged = -1

    while iter < maxit
        iter += 1
        accepted = false

        # solve (J'J + λD'D) * p = J'f
        p, info = __cholsolve!(Size((N, N)), Size((N,)), x, JtJ, λ, DtD, g)

        if info == 0
            # project
            if lb !== nothing
                p = min.(x.-lb, p)
                all(iszero, p) && @goto reject
            end

            if ub !== nothing
                p = max.(x.-ub, p)
                all(iszero, p) && @goto reject
            end

            # evaluate trial step
            xk = x .- p

            fk = fun(xk, data)
            Fk = sum(abs2, fk)
            nfev += 1

            ρ = zeroT
            # ρ = actual reduction / predicted reduction
            #   = (||f(x)|| - ||f(xk)||) / (||f(x)|| - ||f(x) + J(x)p||)
            #   = (1 - Fk/F) / ( ||Jp||/F + 2λ||Dp||/F)
            #   = (1 - Fk/F) / (λ||Dp||/F - <p,g>/F)
            #   = (F - Fk)   / <p, λD'Dp - g>
            if Fk < F
                ac = F - Fk
                pr = dot(p, (@. λ*DtD*p + g))
                ρ = ac / pr
                accepted = true
            end

            # check step size
            if all(>(0), (@. abs(p) < ϵx * (ϵx + abs(x))))
                if accepted
                    x, f, F = xk, fk, Fk
                end
                converged = 2
                break
            end
        end

        if accepted
            # update x = x + p
            x, xk = xk, x

            # update f = f(x + p)
            f, fk = fk, f

            # update objective
            F, Fk = Fk, F

            # check objective
            if abs(ac) < ϵc*Fk && pr < ϵc*Fk && ρ < 2
                converged = 1
                break
            end

            # jacobian J = J(x + p)
            J = jacobian(x, data)
            njev += 1

            # g = J'f
            g = J'*f

            # check gradient
            if norm(g, Inf) < ϵg
                converged = 3
                break
            end

            # update scaling
            JtJ = J'*J
            DtD = clamp.(diag(JtJ), DtD, Dhi)

            # update λ
            if factorupdate === :nielsen
                ν = 2*ρ - 1
                λ *= max(λacc, 1 - (λrej-1)*ν*ν*ν)
                ν = λrej

            else # :marquardt
                if ρ > 0.75
                    λ *= λacc
                elseif ρ < 1e-3
                    λ *= λrej
                end
            end

        else
            @label reject
            # update λ
            if factorupdate === :nielsen
                λ *= ν
                ν *= 2
            else # :marquardt
                λ *= λrej
            end
        end

        λ = clamp(λ, λlo, λhi)
    end

    return x, F, converged, iter, nfev, njev
end


#####
##### Misc
#####

@inline _mul!(Y, A, B) = mul!(Y, A, B)

@inline function _mul!(Y, A::StaticArrays.StaticMatMulLike, B)
    Y .= A * B
    return Y
end

@inline function _mul!(
    Y,
    A::Union{TA, Adjoint{<:Any, TA}, Transpose{<:Any, TA}},
    B::Union{TB, Adjoint{<:Any, TB}, Transpose{<:Any, TB}},
) where {TA<:SparseMatrixCSC, TB<:SparseMatrixCSC}
    Y .= A * B
    return Y
end


@inline function _ldiv!(x, A, b)
    x = copyto!(x, b)
    x = ldiv!(A, x)
    return x
end
