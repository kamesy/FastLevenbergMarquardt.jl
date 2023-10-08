# FastLevenbergMarquardt.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kamesy.github.io/FastLevenbergMarquardt.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kamesy.github.io/FastLevenbergMarquardt.jl/dev/)
[![Build Status](https://github.com/kamesy/FastLevenbergMarquardt.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kamesy/FastLevenbergMarquardt.jl/actions/workflows/CI.yml?query=branch%3Amain)

Levenberg-Marquardt algorithm for solving nonlinear least squares problems.

## Installation

FastLevenbergMarquardt requires Julia v1.6 or later.

```julia
julia> ]add FastLevenbergMarquardt
```

## Usage

With [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl):
```julia
using FastLevenbergMarquardt, StaticArrays

# Beale's function
function f(x, p)
    x1, x2 = x[1], x[2]
    f1 = 1.5   - x1*(1 - x2)
    f2 = 2.25  - x1*(1 - x2*x2)
    f3 = 2.625 - x1*(1 - x2*x2*x2)
    return SVector(f1, f2, f3)
end

function j(x, p)
    x1, x2 = x[1], x[2]
    J11 = -1 + x2
    J21 = -1 + x2*x2
    J31 = -1 + x2*x2*x2
    J12 = x1
    J22 = 2*x1*x2
    J32 = 3*x1*x2*x2
    return SMatrix{3, 2}(J11, J21, J31, J12, J22, J32)
end

x0 = SVector(1.0, 1.0)
x, = lmsolve(f, j, x0)

# pass data to f and j
p  = "some data"
x, = lmsolve(f, j, x0, p)

# add constraints
x0 = SVector(-4.5, 4.5)
lb = SVector(0.5, 0.0)
ub = 4.0
x, = lmsolve(f, j, x0, p, lb, ub)
```

Or in-place:
```julia
function f!(f, x, p)
    x1, x2 = x[1], x[2]
    f[1] = 1.5   - x1*(1 - x2)
    f[2] = 2.25  - x1*(1 - x2*x2)
    f[3] = 2.625 - x1*(1 - x2*x2*x2)
    return f
end

function j!(J, x, p)
    x1, x2 = x[1], x[2]
    J[1,1] = -1 + x2
    J[2,1] = -1 + x2*x2
    J[3,1] = -1 + x2*x2*x2
    J[1,2] = x1
    J[2,2] = 2*x1*x2
    J[3,2] = 3*x1*x2*x2
    return J
end

m, n = (3, 2)
x0 = [1.0, 1.0]
x, = lmsolve!(f!, j!, x0, m)

# pass data to f! and j!
x0 = [1.0, 1.0]
p  = "some data"
x, = lmsolve!(f!, j!, x0, m, p)

# with preallocated arrays
x0 = [1.0, 1.0]
F  = zeros(m)
J  = zeros(m, n)
x, = lmsolve!(f!, j!, x0, F, J, p)

# with preallocated workspace and solver
x0 = [1.0, 1.0]
LM = LMWorkspace(x0, F, J)
solver = CholeskySolver(similar(x0, (n, n)))
x, = lmsolve!(f!, j!, LM, p, solver=solver)

# add constraints
x0 = [1.0, 1.0]
LM.x .= [-4.5, 4.5]
lb = 0.5
ub = SVector(4.0, 0.5)
x, = lmsolve!(f!, j!, LM, p, solver=solver)
```
