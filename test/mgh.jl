##### Jorge More, Burton Garbow, and Kenneth Hillstrom, Testing unconstrained
##### optimization software, ACM Transactions on Mathematical Software,
##### Volume 7, pages 17-41, 1981.


struct MGH{F, J, A, D}
    name::String
    n::Int
    m::Int
    x0::Vector{Float64}
    fx::Vector{Float64}
    x::Vector{Vector{Float64}}
    data::D
    fun!::F
    jac!::J
    avv!::A
end

function MGH(name, n, m, x0, fx, x, data, fun!, jac!, avv! = nothing)
    x0 = Vector{Float64}(x0)
    fx = Vector{Float64}(fx)
    x  = Vector{Float64}.(x)
    return MGH(name, n, m, x0, fx, x, data, fun!, jac!, avv!)
end


#####
##### (1) Rosenbrock
#####
function rosenbrock_f!(f, x, _)
    f[1] = 10*(x[2] - x[1]^2)
    f[2] = 1 - x[1]
    return f
end

function rosenbrock_j!(J, x, _)
    J[1,1] = -20 * x[1]
    J[2,1] = -1
    J[1,2] =  10
    J[2,2] =  0
    return J
end

function rosenbrock_avv!(a, v, x, _)
    a[1] = -20*v[1]*v[1]
    a[2] = 0
    return a
end

function Rosenbrock()
    rosenbrock_n  = 2 # fixed
    rosenbrock_m  = 2 # fixed
    rosenbrock_x0 = [-1.2, 1.0]
    rosenbrock_fx = [0.0]
    rosenbrock_x  = [[1.0, 1.0]]
    rosenbrock_data = nothing

    return MGH(
        "Rosenbrock",
        rosenbrock_n,
        rosenbrock_m,
        rosenbrock_x0,
        rosenbrock_fx,
        rosenbrock_x,
        rosenbrock_data,
        rosenbrock_f!,
        rosenbrock_j!,
        rosenbrock_avv!,
    )
end


#####
##### (2) Freudenstein and Roth
#####
function roth_f!(f, x, _)
    f[1] = -13 + x[1] + ((5 - x[2])*x[2] - 2)*x[2]
    f[2] = -29 + x[1] + ((x[2] + 1)*x[2] - 14)*x[2]
    return f
end

function roth_j!(J, x, _)
    J[1,1] =  1
    J[2,1] =  1
    J[1,2] = (10 - 3*x[2])*x[2] - 2
    J[2,2] = (3*x[2] + 2)*x[2] - 14
    return J
end

function roth_avv!(a, v, x, _)
    a[1] = v[2]*v[2] * (10 - 6*x[2])
    a[2] = v[2]*v[2] * (6*x[2] + 2)
    return a
end

function Roth()
    roth_n  = 2 # fixed
    roth_m  = 2 # fixed
    roth_x0 = [0.5, -2.0]
    roth_fx = [0.0, 48.9842]
    roth_x  = [[5.0, 4.0], [11.4128, -0.896805]]
    roth_data = nothing

    return MGH(
        "Roth",
        roth_n,
        roth_m,
        roth_x0,
        roth_fx,
        roth_x,
        roth_data,
        roth_f!,
        roth_j!,
        roth_avv!,
    )
end


#####
##### (3) Powell badly scaled function
#####
function powell3_f!(f, x, _)
    f[1] = 1e4*x[1]*x[2] - 1
    f[2] = exp(-x[1]) + exp(-x[2]) - 1.0001
    return f
end

function powell3_j!(J, x, _)
    J[1,1] =  1e4*x[2]
    J[2,1] = -exp(-x[1])
    J[1,2] =  1e4*x[1]
    J[2,2] = -exp(-x[2])
    return J
end

function powell3_avv!(a, v, x, _)
    a[1] = 2e4*v[1]*v[2]
    a[2] = v[1]*v[1]*exp(-x[1]) + v[2]*v[2]*exp(-x[2])
    return a
end

function Powell3()
    powell3_n  = 2 # fixed
    powell3_m  = 2 # fixed
    powell3_x0 = [0.0, 1.0]
    powell3_fx = [0.0]
    powell3_x  = [[1.098159e-5, 9.106146]]
    powell3_data = nothing

    return MGH(
        "Powell3",
        powell3_n,
        powell3_m,
        powell3_x0,
        powell3_fx,
        powell3_x,
        powell3_data,
        powell3_f!,
        powell3_j!,
        powell3_avv!,
    )
end


#####
##### (4) Brown badly scaled function
#####
function brown4_f!(f, x, _)
    f[1] = x[1] - 1e6
    f[2] = x[2] - 2e-6
    f[3] = x[1]*x[2] - 2
    return f
end

function brown4_j!(J, x, _)
    J[1,1] = 1
    J[2,1] = 0
    J[3,1] = x[2]
    J[1,2] = 0
    J[2,2] = 1
    J[3,2] = x[1]
    return J
end

function brown4_avv!(a, v, x, _)
    a[1] = 0
    a[2] = 0
    a[3] = 2*v[1]*v[2]
    return a
end

function Brown4()
    brown4_n  = 2 # fixed
    brown4_m  = 3 # fixed
    brown4_x0 = [1.0, 1.0]
    brown4_fx = [0.0]
    brown4_x  = [[1e6, 2e-6]]
    brown4_data = nothing

    return MGH(
        "Brown4",
        brown4_n,
        brown4_m,
        brown4_x0,
        brown4_fx,
        brown4_x,
        brown4_data,
        brown4_f!,
        brown4_j!,
        brown4_avv!,
    )
end


#####
##### (5) Beale function
#####
function beale_f!(f, x, _)
    f[1] = 1.5   - x[1]*(1 - x[2])
    f[2] = 2.25  - x[1]*(1 - x[2]^2)
    f[3] = 2.625 - x[1]*(1 - x[2]^3)
    return f
end

function beale_j!(J, x, _)
    J[1,1] = -1 + x[2]
    J[2,1] = -1 + x[2]^2
    J[3,1] = -1 + x[2]^3
    J[1,2] =   x[1]
    J[2,2] = 2*x[1]*x[2]
    J[3,2] = 3*x[1]*x[2]^2
    return J
end

function beale_avv!(a, v, x, _)
    a[1] = 2*v[1]*v[2]
    a[2] = 2*v[1]*v[2]*2*x[2]   + v[2]*v[2]*2*x[1]
    a[3] = 2*v[1]*v[2]*3*x[2]^2 + v[2]*v[2]*6*x[1]*x[2]
    return a
end

function Beale()
    beale_n  = 2 # fixed
    beale_m  = 3 # fixed
    beale_x0 = [1.0, 1.0]
    beale_fx = [0.0]
    beale_x  = [[3.0, 0.5]]
    beale_data = nothing

    return MGH(
        "Beale",
        beale_n,
        beale_m,
        beale_x0,
        beale_fx,
        beale_x,
        beale_data,
        beale_f!,
        beale_j!,
        beale_avv!,
    )
end


#####
##### (6) Jennrich and Sampson function
#####
function jennrich_f!(f, x, _)
    x1, x2 = x[1], x[2]
    for i in eachindex(f)
        f[i] = 2 + 2*i - exp(i*x1) - exp(i*x2)
    end
    return f
end

function jennrich_j!(J, x, _)
    x1, x2 = x[1], x[2]
    for i in axes(J, 1)
        J[i,1] = -i*exp(i*x1)
        J[i,2] = -i*exp(i*x2)
    end
    return J
end

function Jennrich()
    jennrich_n  = 2  # fixed
    jennrich_m  = 10 # fixed
    jennrich_x0 = [0.3, 0.4]
    jennrich_fx = [124.362]
    jennrich_x  = [[0.257825, 0.257825]]
    jennrich_data = nothing

    return MGH(
        "Jennrich",
        jennrich_n,
        jennrich_m,
        jennrich_x0,
        jennrich_fx,
        jennrich_x,
        jennrich_data,
        jennrich_f!,
        jennrich_j!,
    )
end


#####
##### (7) Helical valley function
#####
function helical_f!(f, x, _)
    f[1] = 10*(x[3] - 5/pi*atan(x[2], x[1]))
    f[2] = 10*(hypot(x[1], x[2]) - 1)
    f[3] = x[3]
    return f
end

function helical_j!(J, x, _)
    c = hypot(x[1], x[2])
    c2 = c * c

    J[1,1] =  50/pi*x[2]/c2
    J[2,1] =  10*x[1]/c
    J[3,1] =  0

    J[1,2] = -50/pi*x[1]/c2
    J[2,2] =  10*x[2]/c
    J[3,2] =  0

    J[1,3] =  10
    J[2,3] =  0
    J[3,3] =  1

    return J
end

function helical_avv!(a, v, x, _)
    c = hypot(x[1], x[2])
    a[1] = 100/pi * (-v[1]*v[1]*x[1]*x[2] + v[1]*v[2]*(x[1]^2-x[2]^2) + v[2]*v[2]*x[1]*x[2]) / c^4
    a[2] = 10*(v[1]*x[2]-v[2]*x[1])^2 / c^3
    a[3] = 0
    return a
end

function Helical()
    helical_n  = 3 # fixed
    helical_m  = 3 # fixed
    helical_x0 = [-1.0, 0.0, 0.0]
    helical_fx = [0.0]
    helical_x  = [[1.0, 0.0, 0.0]]
    helical_data = nothing

    return MGH(
        "Helical",
        helical_n,
        helical_m,
        helical_x0,
        helical_fx,
        helical_x,
        helical_data,
        helical_f!,
        helical_j!,
        helical_avv!,
    )
end


#####
##### (8) Bard function
#####
function bard_f!(f, x, y)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(f)
        ui, vi = i, 16-i
        wi = min(ui, vi)
        f[i] = y[i] - (x1 + ui/(vi*x2 + wi*x3))
    end
    return f
end

function bard_j!(J, x, y)
    x2, x3 = x[2], x[3]
    for i in axes(J, 1)
        ui, vi = i, 16-i
        wi = min(ui, vi)

        den = vi*x2 + wi*x3
        den *= den

        J[i,1] = -1
        J[i,2] = ui*vi / den
        J[i,3] = ui*wi / den
    end
    return J
end

function Bard()
    bard_n  = 3  # fixed
    bard_m  = 15 # fixed
    bard_x0 = [1.0, 1.0, 1.0]
    bard_fx = [8.21487e-3, 17.4286]
    bard_x  = [[0.082411, 1.133036, 2.343695], [0.8406, -Inf, -Inf]]
    bard_data = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]

    return MGH(
        "Bard",
        bard_n,
        bard_m,
        bard_x0,
        bard_fx,
        bard_x,
        bard_data,
        bard_f!,
        bard_j!,
    )
end


#####
##### (9) Gaussian function
#####
function gaussian_f!(f, x, y)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(f)
        t = (8 - i) / 2
        z = t - x3
        f[i] = x1 * exp(-0.5*x2*z*z) - y[i]
    end
    return f
end

function gaussian_j!(J, x, y)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(y)
        t = (8 - i) / 2
        z = t - x3
        z2 = z * z
        ex = exp(-0.5*x2*z2)

        J[i,1] =    ex
        J[i,2] = x1*ex * -0.5*z2
        J[i,3] = x1*ex * x2*z
    end
    return J
end

function Gaussian()
    gaussian_n  = 3  # fixed
    gaussian_m  = 15 # fixed
    gaussian_x0 = [0.4, 1.0, 0.0]
    gaussian_fx = [1.12793e-8]
    gaussian_x  = [[]]
    gaussian_data = [0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989, 0.3521, 0.2420, 0.1295, 0.0540, 0.0175, 0.0044, 0.0009]

    return MGH(
        "Gaussian",
        gaussian_n,
        gaussian_m,
        gaussian_x0,
        gaussian_fx,
        gaussian_x,
        gaussian_data,
        gaussian_f!,
        gaussian_j!,
    )
end


#####
##### (10) Meyer function
#####
function meyer_f!(f, x, y)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(y)
        t = 45 + 5*i
        f[i] = x1 * exp(x2/(t + x3)) - y[i]
    end
    return f
end

function meyer_j!(J, x, y)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(y)
        t = 45 + 5*i
        z = t + x3
        z2 = z * z
        ex = exp(x2 / z)

        J[i,1] =    ex
        J[i,2] = x1*ex / z
        J[i,3] = x1*ex * -x2/z2
    end
    return J
end

function Meyer()
    meyer_n  = 3  # fixed
    meyer_m  = 16 # fixed
    meyer_x0 = [0.02, 4000.0, 250.0]
    meyer_fx = [87.9458]
    meyer_x = [[0.00560964, 6181.35, 345.224]]
    meyer_data = [34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872]

    return MGH(
        "Meyer",
        meyer_n,
        meyer_m,
        meyer_x0,
        meyer_fx,
        meyer_x,
        meyer_data,
        meyer_f!,
        meyer_j!,
    )
end


#####
##### (11) Gulf research and development function
#####
function gulf_f!(f, x, _)
    m = length(f)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in 1:m
        mi = m*i
        ti = i/100
        yi = 25 + (-50*log(ti))^2/3
        f[i] = exp(-(abs(yi*mi*x2)^x3)/x1) - ti
    end
    return f
end

function gulf_j!(J, x, _)
    m, n = size(J)
    x1, x2, x3 = x[1], x[2], x[3]
    x12 = x1 * x1
    for i in 1:m
        mi = m*i
        ti = i/100
        yi = 25 + (-50*log(ti))^2/3

        c = mi * x2 * yi
        z = abs(c)
        zx3 = z^x3

        ex = exp(-zx3 / x1)

        J[i,1] =  (zx3 * ex) / x12
        J[i,2] = -(mi*x3*yi*zx3/z * sign(c) * ex) / x1
        J[i,3] = -(zx3*ex*conj(log(z))) / x1
    end
    return J
end

function Gulf(m::Int = 20)
    @assert 3 <= m <= 100

    gulf_n  = 3 # fixed
    gulf_m  = m # n <= m <= 100
    gulf_x0 = [5.0, 2.5, 0.15]
    gulf_fx = [0.0]
    gulf_x  = [[50.0, 25.0, 1.5]]
    gulf_data = nothing

    return MGH(
        "Gulf-$m",
        gulf_n,
        gulf_m,
        gulf_x0,
        gulf_fx,
        gulf_x,
        gulf_data,
        gulf_f!,
        gulf_j!,
    )
end


#####
##### (12) Box three-dimensional function
#####
function box3d_f!(f, x, _)
    x1, x2, x3 = x[1], x[2], x[3]
    for i in eachindex(f)
        ti = 0.1*i
        f[i] = exp(-ti*x1) - exp(-ti*x2) - x3*(exp(-ti) - exp(-10*ti))
    end
    return f
end

function box3d_j!(J, x, _)
    x1, x2 = x[1], x[2]
    for i in axes(J, 1)
        ti = 0.1*i
        J[i,1] = -ti*exp(-ti*x1)
        J[i,2] =  ti*exp(-ti*x2)
        J[i,3] = -(exp(-ti) - exp(-10*ti))
    end
    return J
end

function Box3d(m::Int = 10)
    @assert m >= 3

    box3d_n  = 3 # fixed
    box3d_m  = m # m >= n
    box3d_x0 = [0.0, 10.0, 20.0]
    box3d_fx = [0.0]
    box3d_x  = [[]] # [[1.0, 10.0, 1.0], [10.0, 1.0, -1.0], [x1 = x2, 0.0]]
    box3d_data = nothing

    return MGH(
        "Box3d-$m",
        box3d_n,
        box3d_m,
        box3d_x0,
        box3d_fx,
        box3d_x,
        box3d_data,
        box3d_f!,
        box3d_j!,
    )
end


#####
##### (13) Powell singular function
#####
function powell13_f!(f, x, _)
    f[1] = x[1] + 10*x[2]
    f[2] = sqrt(5)*(x[3] - x[4])
    f[3] = (x[2] - 2*x[3])^2
    f[4] = sqrt(10)*(x[1] - x[4])^2
    return f
end

function powell13_j!(J, x, _)
    sqrt5 = sqrt(5)
    sqrt10 = sqrt(10)

    c14 = x[1] - x[4]
    c23 = x[2] - 2*x[3]

    J[1,1] = 1
    J[2,1] = 0
    J[3,1] = 0
    J[4,1] = 2*sqrt10*c14

    J[1,2] = 10
    J[2,2] = 0
    J[3,2] = 2*c23
    J[4,2] = 0

    J[1,3] = 0
    J[2,3] = sqrt5
    J[3,3] = -4*c23
    J[4,3] = 0

    J[1,4] = 0
    J[2,4] = -sqrt5
    J[3,4] = 0
    J[4,4] = -2*sqrt10*c14
    return J
end

function powell13_avv!(a, v, x, _)
    a[1] = 0
    a[2] = 0
    a[3] = v[2]*v[2]*2 - 8*v[2]*v[3] + v[3]*v[3]*8
    a[4] = v[1]*v[1]*2*sqrt(10) - 2*v[1]*v[4]*2*sqrt(10) + v[4]*v[4]*2*sqrt(10)
    return a
end

function Powell13()
    powell13_n  = 4 # fixed
    powell13_m  = 4 # fixed
    powell13_x0 = [3.0, -1.0, 0.0, 1.0]
    powell13_fx = [0.0]
    powell13_x  = [[0.0, 0.0, 0.0, 0.0]]
    powell13_data = nothing

    return MGH(
        "Powell13",
        powell13_n,
        powell13_m,
        powell13_x0,
        powell13_fx,
        powell13_x,
        powell13_data,
        powell13_f!,
        powell13_j!,
        powell13_avv!,
    )
end


#####
##### (14) Wood function
#####
function wood_f!(f, x, _)
    sqrt10 = sqrt(10)
    f[1] = 10*(x[2] - x[1]^2)
    f[2] = 1 - x[1]
    f[3] = 3*sqrt10*(x[4] - x[3]^2)
    f[4] = 1 - x[3]
    f[5] = sqrt10*(x[2] + x[4] - 2)
    f[6] = (x[2] - x[4])/sqrt10
    return f
end

function wood_j!(J, x, _)
    sqrt10 = sqrt(10)
    fill!(J, 0)
    J[1,1] = -20 * x[1]
    J[2,1] = -1
    J[1,2] =  10
    J[5,2] =  sqrt10
    J[6,2] =  1/sqrt10
    J[3,3] = -6 * sqrt10 * x[3]
    J[4,3] = -1
    J[3,4] =  3*sqrt10
    J[5,4] =  sqrt10
    J[6,4] = -1/sqrt10
    return J
end

function wood_avv!(a, v, x, _)
    fill!(a, 0)
    a[1] = -20*v[1]*v[1]
    a[3] = -6*sqrt(10)*v[3]*v[3]
    return a
end

function Wood()
    wood_n  = 4 # fixed
    wood_m  = 6 # fixed
    wood_x0 = [-3.0, -1.0, -3.0, -1.0]
    wood_fx = [0.0]
    wood_x  = [[1.0, 1.0, 1.0, 1.0]]
    wood_data = nothing

    return MGH(
        "Wood",
        wood_n,
        wood_m,
        wood_x0,
        wood_fx,
        wood_x,
        wood_data,
        wood_f!,
        wood_j!,
        wood_avv!,
    )
end


#####
##### (15) Kowalik and Osborne function
#####
function kowalik_f!(f, x, (y, u))
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    for i in eachindex(f)
        ui = u[i]
        den = @evalpoly(ui, x4, x3, 1)
        f[i] = y[i] - x1*ui*(ui + x2) / den
    end
    return f
end

function kowalik_j!(J, x, (y, u))
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    for i in eachindex(y)
        ui = u[i]
        num = ui*(ui + x2)
        den = @evalpoly(ui, x4, x3, 1)
        den2 = den * den

        J[i,1] =      -num / den
        J[i,2] =    -x1*ui / den
        J[i,3] = ui*x1*num / den2
        J[i,4] =    x1*num / den2
    end
    return J
end

function kowalik_avv!(a, v, x, (y, u))
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    v1, v2, v3, v4 = v[1], v[2], v[3], v[4]
    @. a = (2*u*(v1*x4 - v4*x1 + u^2*v1 + u*v1*x3 - u*v3*x1)*(u*v4 - v2*x4 + v4*x2 - u^2*v2 + u^2*v3 - u*v2*x3 + u*v3*x2))/(u^2 + x3*u + x4)^3
    return a
end

function Kowalik()
    kowalik_n  = 4  # fixed
    kowalik_m  = 11 # fixed
    kowalik_x0 = [0.25, 0.39, 0.415, 0.39]
    kowalik_fx = [3.07505e-4, 1.02734e-3]
    kowalik_x  = [[0.192807, 0.191282, 0.123057, 0.136062], [Inf, -14.07, -Inf, -Inf]]
    kowalik_data = (
        [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246],
        [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625],
    )

    return MGH(
        "Kowalik",
        kowalik_n,
        kowalik_m,
        kowalik_x0,
        kowalik_fx,
        kowalik_x,
        kowalik_data,
        kowalik_f!,
        kowalik_j!,
        kowalik_avv!,
    )
end


#####
##### (16) Brown and Dennis function
#####
function dennis_f!(f, x, _)
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    for i in eachindex(f)
        ti = i/5
        a = x1 + x2*ti - exp(ti)
        b = x3 + x4*sin(ti) - cos(ti)
        f[i] = a*a + b*b
    end
    return f
end

function dennis_j!(J, x, _)
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    for i in axes(J, 1)
        ti = i/5
        a = x1 + x2*ti - exp(ti)
        b = x3 + x4*sin(ti) - cos(ti)

        J[i,1] = 2*a
        J[i,2] = 2*ti*a
        J[i,3] = 2*b
        J[i,4] = 2*sin(ti)*b
    end
    return J
end

function Dennis()
    dennis_n  = 4  # fixed
    dennis_m  = 20 # fixed
    dennis_x0 = [25.0, 5.0, -5.0, -1.0]
    dennis_fx = [85822.2]
    dennis_x  = [[-11.5944, 13.2036, -0.4034, 0.2368]]
    dennis_data = nothing

    return MGH(
        "Dennis",
        dennis_n,
        dennis_m,
        dennis_x0,
        dennis_fx,
        dennis_x,
        dennis_data,
        dennis_f!,
        dennis_j!,
    )
end


#####
##### (17) Osborne 1 function
#####
function osborne1_f!(f, x, y)
    x1, x2, x3, x4, x5 = x[1], x[2], x[3], x[4], x[5]
    for i in eachindex(y)
        ti = 10*(i-1)
        f[i] = y[i] - (x1 + x2*exp(-ti*x4) + x3*exp(-ti*x5))
    end
    return f
end

function osborne1_j!(J, x, y)
    x2, x3, x4, x5 = x[2], x[3], x[4], x[5]
    for i in eachindex(y)
        ti = 10*(i-1)
        a = exp(-ti*x4)
        b = exp(-ti*x5)
        J[i,1] = -1
        J[i,2] = -a
        J[i,3] = -b
        J[i,4] = ti*x2*a
        J[i,5] = ti*x3*b
    end
    return J
end

function Osborne1()
    osborne1_n  = 5  # fixed
    osborne1_m  = 33 # fixed
    osborne1_x0 = [0.5, 1.5, -1.0, 0.01, 0.02]
    osborne1_fx = [5.46489e-5]
    osborne1_x  = [[0.37541, 1.93585, -1.46469, 0.01287, 0.02212]]
    osborne1_data = [0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406]

    return MGH(
        "Osborne1",
        osborne1_n,
        osborne1_m,
        osborne1_x0,
        osborne1_fx,
        osborne1_x,
        osborne1_data,
        osborne1_f!,
        osborne1_j!,
    )
end


#####
##### (18) Biggs EXP6 function
#####
function biggs_f!(f, x, y)
    x1, x2, x3, x4, x5, x6 = x[1], x[2], x[3], x[4], x[5], x[6]
    for i in eachindex(y)
        ti = 0.1*i
        f[i] = x3*exp(-ti*x1) - x4*exp(-ti*x2) + x6*exp(-ti*x5) - y[i]
    end
    return f
end

function biggs_j!(J, x, y)
    x1, x2, x3, x4, x5, x6 = x[1], x[2], x[3], x[4], x[5], x[6]
    for i in eachindex(y)
        ti = 0.1*i
        a = exp(-ti*x1)
        b = exp(-ti*x2)
        c = exp(-ti*x5)

        J[i,1] = -ti*x3*a
        J[i,2] =  ti*x4*b
        J[i,3] =  a
        J[i,4] = -b
        J[i,5] = -ti*x6*c
        J[i,6] =  c
    end
    return J
end

function Biggs(m::Int = 13)
    @assert m >= 6

    biggs_n  = 6 # fixed
    biggs_m  = m # m >= n
    biggs_x0 = [1, 2, 1, 1, 1, 1]
    biggs_fx = [0.0]
    biggs_x  = [[]]
    biggs_data = [exp(-0.1*i) - 5*exp(-i) + 3*exp(-0.4*i) for i in 1:m]

    return MGH(
        "Biggs-$m",
        biggs_n,
        biggs_m,
        biggs_x0,
        biggs_fx,
        biggs_x,
        biggs_data,
        biggs_f!,
        biggs_j!,
    )
end


#####
##### (19) Osborne 2 function
#####
function osborne2_f!(f, x, y)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]
    for i in eachindex(f)
        ti = (i - 1) / 10
        z1 = ti * x5
        z2 = ti - x9
        z3 = ti - x10
        z4 = ti - x11
        f[i] = y[i] - (x1*exp(-z1) + x2*exp(-z2*z2*x6) + x3*exp(-z3*z3*x7) + x4*exp(-z4*z4*x8))
    end
    return f
end

function osborne2_j!(J, x, y)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]
    for i in eachindex(y)
        ti = (i - 1) / 10

        z1 = ti - x9
        z2 = ti - x10
        z3 = ti - x11

        z12 = z1 * z1
        z22 = z2 * z2
        z32 = z3 * z3

        ex0 = exp(-ti*x5)
        ex1 = exp(-z12*x6)
        ex2 = exp(-z22*x7)
        ex3 = exp(-z32*x8)

        J[i,1]  =         -ex0
        J[i,2]  =         -ex1
        J[i,3]  =         -ex2
        J[i,4]  =         -ex3
        J[i,5]  =    ti*x1*ex0
        J[i,6]  =       x2*ex1*z12
        J[i,7]  =       x3*ex2*z22
        J[i,8]  =       x4*ex3*z32
        J[i,9]  = -2*x6*x2*ex1*z1
        J[i,10] = -2*x7*x3*ex2*z2
        J[i,11] = -2*x8*x4*ex3*z3
    end
    return J
end

function Osborne2()
    osborne2_n  = 11 # fixed
    osborne2_m  = 65 # fixed
    osborne2_x0 = [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5]
    osborne2_fx = [4.01377e-2]
    osborne2_x  = [[]]
    osborne2_data = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]

    return MGH(
        "Osborne2",
        osborne2_n,
        osborne2_m,
        osborne2_x0,
        osborne2_fx,
        osborne2_x,
        osborne2_data,
        osborne2_f!,
        osborne2_j!,
    )
end


#####
##### (20) Watson function
#####
function watson_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == 31

    oneT = one(eltype(x))
    zeroT = zero(eltype(x))

    for i in 1:29
        ti = i/29
        a  = zeroT
        b  = x[1]

        da = oneT
        db = ti

        for j in 2:n
            xj  = x[j]
            a  += (j-1)*da*xj
            b  += db*xj
            da  = db
            db *= ti
        end

        f[i] = a - b*b - 1
    end

    f[30] = x[1]
    f[31] = x[2] - x[1]*x[1] - 1

    return f
end

function watson_j!(J, x, _)
    m, n = size(J)
    @assert m == 31

    oneT = one(eltype(x))
    zeroT = zero(eltype(x))

    for i in 1:29
        ti = i/29

        b = zeroT
        db = oneT
        for j in 1:n
            b += db*x[j]
            db *= ti
        end

        da = oneT
        db = ti
        J[i,1] = -2*b

        for j in 2:n
            J[i,j] = (j-1)*da - 2*db*b
            da  = db
            db *= ti
        end
    end

    for j in 1:n
        J[30,j] = 0
        J[31,j] = 0
    end

    J[30,1] =  1
    J[31,1] = -2*x[1]
    J[31,2] =  1

    return J
end

function Watson(n::Int = 6)
    @assert n ∈ (6, 9, 12)

    watson_n  = n
    watson_m  = 31 # fixed
    watson_x0 = zeros(n)
    watson_fx = [n == 6 ? 2.28767e-3 : n == 9 ? 1.39976e-6 : 4.72238e-10]
    watson_x  = [[]]
    watson_data = nothing

    return MGH(
        "Watson-$n",
        watson_n,
        watson_m,
        watson_x0,
        watson_fx,
        watson_x,
        watson_data,
        watson_f!,
        watson_j!,
    )
end


#####
##### (21) Extended Rosenbrock function
#####
function rosenbrockn_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n && n % 2 == 0
    for i in 1:2:n
        x1, x2 = x[i], x[i+1]
        f[i]   = 10*(x2 - x1*x1)
        f[i+1] = 1 - x1
    end
    return f
end

function rosenbrockn_j!(J, x, _)
    m, n = size(J)
    @assert m == n && n % 2 == 0
    fill!(J, 0)
    for i in 1:2:n
        J[i,i] = -20*x[i]
        J[i,i+1] = 10
        J[i+1,i] = -1
    end
    return J
end

function RosenbrockN(n::Int = 24)
    @assert n >= 1
    @assert iseven(n)

    rosenbrock_n  = n # even
    rosenbrock_m  = n # m = n
    rosenbrock_x0 = [ifelse(isodd(i), -1.2, 1.0) for i in 1:n]
    rosenbrock_fx = [0.0]
    rosenbrock_x  = [ones(n)]
    rosenbrock_data = nothing

    return MGH(
        "RosenbrockN-$n",
        rosenbrock_n,
        rosenbrock_m,
        rosenbrock_x0,
        rosenbrock_fx,
        rosenbrock_x,
        rosenbrock_data,
        rosenbrockn_f!,
        rosenbrockn_j!,
    )
end


#####
##### (22) Extended Powell singular function
#####
function powell22_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n && n % 4 == 0

    sqrt5 = sqrt(5)
    sqrt10 = sqrt(10)

    for i in 1:4:n
        x1, x2, x3, x4 = x[i], x[i+1], x[i+2], x[i+3]

        a = x3 - x4
        b = x2 - 2*x3
        c = x1 - x4

        f[i]   = x1 + 10*x2
        f[i+1] = sqrt5*a
        f[i+2] = b*b
        f[i+3] = sqrt10*c*c
    end

    return f
end

function powell22_j!(J, x, _)
    m, n = size(J)
    @assert m == n && n % 4 == 0
    fill!(J, 0)

    sqrt5 = sqrt(5)
    twosqrt10 = 2*sqrt(10)

    for i in 1:4:n
        x1, x2, x3, x4 = x[i], x[i+1], x[i+2], x[i+3]

        b = x2 - 2*x3
        c = x1 - x4

        J[i,i]   = 1
        J[i,i+1] = 10

        J[i+1,i+2] =  sqrt5
        J[i+1,i+3] = -sqrt5

        J[i+2,i+1] =  2*b
        J[i+2,i+2] = -4*b

        J[i+3,i]   =  twosqrt10*c
        J[i+3,i+3] = -twosqrt10*c
    end

    return J
end

function Powell22(n::Int = 16)
    @assert n >= 1
    @assert n % 4 == 0

    powell22_n  = n # multiple of 4
    powell22_m  = n # m = n
    powell22_x0 = [(3.0, -1.0, 0.0, 1.0)[mod1(j, 4)] for j in 1:n]
    powell22_fx = [0.0]
    powell22_x  = [zeros(n)]
    powell22_data = nothing

    Powell22 = MGH(
        "Powell22-$n",
        powell22_n,
        powell22_m,
        powell22_x0,
        powell22_fx,
        powell22_x,
        powell22_data,
        powell22_f!,
        powell22_j!,
    )
end


#####
##### (23) Penalty function 1
#####
function penalty1_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n+1

    a = sqrt(1e-5)
    b = zero(eltype(x))
    for i in 1:n
        xi = x[i]
        b += xi*xi
        f[i] = a*xi - a
    end
    f[n+1] = b - 0.25

    return f
end

function penalty1_j!(J, x, _)
    m, n = size(J)
    @assert m == n+1

    a = sqrt(1e-5)
    for j in 1:n
        xj = x[j]
        for i in 1:j-1
            J[i,j] = 0
        end
        J[j,j] = a
        for i in j+1:n
            J[i,j] = 0
        end
        J[n+1,j] = 2*xj
    end

    return J
end

function Penalty1(n::Int = 4)
    @assert n ∈ (4, 10)

    penalty1_n  = n # variable
    penalty1_m  = n + 1 # m = n + 1
    penalty1_x0 = collect(1:n)
    penalty1_fx = [n == 4 ? 2.24997e-5 : 7.08765e-5]
    penalty1_x  = [[]]
    penalty1_data = nothing

    return MGH(
        "Penalty1-$n",
        penalty1_n,
        penalty1_m,
        penalty1_x0,
        penalty1_fx,
        penalty1_x,
        penalty1_data,
        penalty1_f!,
        penalty1_j!,
    )
end


#####
##### (24) Penalty function 2
#####
function penalty2_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == 2*n

    a = sqrt(1e-5)
    b = exp(-1/10)

    x1 = x[1]
    f2n = n * x1*x1
    f[1] = x1 - 0.2

    x1 /= 10
    i1 = 1/10

    for i in 2:n
        xi = x[i]
        i0, i1 = i1, i/10
        x0, x1 = x1, xi/10
        y = exp(i1) + exp(i0)
        f[i] = a*(exp(x1) + exp(x0) - y)
        f2n += (n - i + 1) * xi*xi
    end

    for i in n+1:2*n-1
        f[i] = a*(exp(x[i-n+1]/10) - b)
    end

    f[2*n] = f2n - 1

    return f
end

function penalty2_j!(J, x, _)
    m, n = size(J)
    @assert m == 2*n
    fill!(J, 0)

    a = sqrt(1e-5)/10

    J[1,1] = 1
    J[m,1] = 2*n*x[1]
    x1 = x[1]/10

    for j in 2:n
        xj = x[j]
        x0, x1 = x1, xj/10

        J[j,j-1] = a*exp(x0)
        J[j,j]   = a*exp(x1)

        J[n+j-1,j] = a*exp(x1)
        J[m,j] = 2*(n-j+1)*xj
    end

    return J
end

function Penalty2(n::Int = 4)
    @assert n ∈ (4, 10)

    penalty2_n  = n # variable
    penalty2_m  = 2*n # m = 2*n
    penalty2_x0 = fill(0.5, n)
    penalty2_fx = [n == 4 ? 9.37629e-6 : 2.93660e-4]
    penalty2_x  = [[]]
    penalty2_data = nothing

    return MGH(
        "Penalty2-$n",
        penalty2_n,
        penalty2_m,
        penalty2_x0,
        penalty2_fx,
        penalty2_x,
        penalty2_data,
        penalty2_f!,
        penalty2_j!,
    )
end


#####
##### (25) Variably dimensioned function
#####
function vardim_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n + 2
    s = zero(eltype(x))
    for i in 1:n
        xi1 = x[i] - 1
        s += i*xi1
        f[i] = xi1
    end
    f[n+1] = s
    f[n+2] = s*s
    return f
end

function vardim_j!(J, x, _)
    m, n = size(J)
    @assert m == n + 2
    s = zero(eltype(x))
    for i in 1:n
        s += i*(x[i] - 1)
    end
    twos = 2*s
    for j in 1:n
        for i in 1:j-1
            J[i,j] = 0
        end
        J[j,j] = 1
        for i in j+1:n
            J[i,j] = 0
        end
        J[n+1,j] = j
        J[n+2,j] = j*twos
    end
    return J
end

function Vardim(n::Int = 10)
    @assert n >= 1

    vardim_n  = n # variable
    vardim_m  = n + 2 # m = n + 2
    vardim_x0 = collect(1.0 .- (1:n)./n)
    vardim_fx = [0.0]
    vardim_x  = [ones(n)]
    vardim_data = nothing

    return MGH(
        "Vardim-$n",
        vardim_n,
        vardim_m,
        vardim_x0,
        vardim_fx,
        vardim_x,
        vardim_data,
        vardim_f!,
        vardim_j!,
    )
end


#####
##### (26) Trigonometric function
#####
function trig_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n
    c = zero(eltype(x))
    for i in 1:n
        c += cos(x[i])
    end
    c = n - c
    for i in 1:n
        xi = x[i]
        sx, cx = sincos(xi)
        f[i] = c + i*(1 - cx) - sx
    end
    return f
end

function trig_j!(J, x, _)
    m, n = size(J)
    @assert m == n
    for j in 1:n
        for i in 1:j-1
            J[i,j] = sin(x[i])
        end
        sx, cx = sincos(x[j])
        J[j,j] = sx + j*sx - cx
        for i in j+1:n
            J[i,j] = sin(x[i])
        end
    end
    return J
end

function Trig(n::Int = 10)
    @assert n >= 1

    trig_n  = n # variable
    trig_m  = n # m = n
    trig_x0 = fill(1/n, n)
    trig_fx = [0.0]
    trig_x  = [[]]
    trig_data = nothing

    return MGH(
        "Trig-$n",
        trig_n,
        trig_m,
        trig_x0,
        trig_fx,
        trig_x,
        trig_data,
        trig_f!,
        trig_j!,
    )
end


#####
##### (27) Brown almost-linear function
#####
function brown27_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n

    s = zero(eltype(x))
    p = one(eltype(x))

    for i in 1:n
        s += x[i]
        p *= x[i]
    end

    s -= (n + 1)

    for i in 1:n-1
        f[i] = x[i] + s
    end

    f[n] = p - 1

    return f
end

function brown27_j!(J, x, _)
    m, n = size(J)
    @assert m == n

    oneT = one(eltype(x))
    p = oneT

    for j in 1:n
        p *= x[j]
        for i in 1:n
            J[i,j] = 1
        end
        J[j,j] += 1
    end

    for j in 1:n
        c = x[j]
        if c != 0
            J[n,j] = p / c
        else
            c = oneT
            q = oneT
            for k in 1:n
                if k != j
                    q *= x[k]
                end
            end
            J[n,j] = q
        end
    end

    return J
end

function Brown27(n::Int = 10)
    @assert n >= 1

    brown27_n  = n # variable
    brown27_m  = n # m = n
    brown27_x0 = fill(1/2, n)
    brown27_fx = [0.0, 1.0]
    brown27_x  = [[], [ifelse(i < n, 0, n + 1) for i in 1:n]] # f = 0 @ (a, ..., a, a^(1-n)), na^n - (n+1)a^(n-1) + 1 = 0
    brown27_data = nothing

    return MGH(
        "Brown27-$n",
        brown27_n,
        brown27_m,
        brown27_x0,
        brown27_fx,
        brown27_x,
        brown27_data,
        brown27_f!,
        brown27_j!,
    )
end


#####
##### (28) Discrete boundary value function
#####
function dbv_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n

    h = 1 / (n + 1)
    h2 = h * h

    xi, xr = zero(eltype(x)), x[1]

    for i in 1:n-1
        t = i*h
        xl, xi, xr = xi, xr, x[i+1]
        c = xi + t + 1
        f[i] = 2*xi - xl - xr + h2*c*c*c/2
    end

    xl, xi = xi, xr
    c = xi + n*h + 1
    f[n] = 2*xi - xl + h2*c*c*c/2

    return f
end

function dbv_j!(J, x, _)
    m, n = size(J)
    @assert m == n

    h = 1 / (n + 1)
    h2 = h * h

    c = x[1] + h + 1
    J[1,1] = 2 + 3*h2*c*c/2
    J[1,2] = -1
    for i in 3:n
        J[i,1] = 0
    end

    for j in 2:n-1
        c = x[j] + j*h + 1

        for i in 1:j-2
            J[i,j] = 0
        end

        J[j,j-1] = -1
        J[j,j] = 2 + 3*h2*c*c/2
        J[j,j+1] = -1

        for i in j+2:n
            J[i,j] = 0
        end
    end

    c = x[n] + n*h + 1
    for j in 1:n-2
        J[j,n] = 0
    end
    J[n,n-1] = -1
    J[n,n] = 2 + 3*h2*c*c/2

    return J
end

function DBV(n::Int = 10)
    @assert n >= 1

    dbv_n  = n # variable
    dbv_m  = n # m = n
    dbv_x0 = [(ti = i/(n+1); ti*(ti-1)) for i in 1:n]
    dbv_fx = [0.0]
    dbv_x  = [[]]
    dbv_data = nothing

    return MGH(
        "DBV-$n",
        dbv_n,
        dbv_m,
        dbv_x0,
        dbv_fx,
        dbv_x,
        dbv_data,
        dbv_f!,
        dbv_j!,
    )
end


#####
##### (29) Discrete integral equation function
#####
function dint_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n
    zeroT = zero(eltype(x))

    h = 1 / (n + 1)

    for i in 1:n
        xi = x[i]
        ti = i*h
        a = zeroT
        b = zeroT
        for j in 1:i
            tj = j*h
            c = x[j] + tj + 1
            a += tj*c*c*c
        end
        a *= (1 - ti)
        for j in i+1:n
            tj = j*h
            c = x[j] + tj + 1
            b += (1 - tj)*c*c*c
        end
        b *= ti
        f[i] = xi + h*(a + b)/2
    end

    return f
end

function dint_j!(J, x, _)
    m, n = size(J)
    @assert m == n
    h = 1 / (n + 1)

    for j in 1:n
        tj = j*h
        xj = x[j]

        c = xj + tj + 1
        a = 3*tj*c*c
        b = 3*(1-tj)*c*c

        for i in 1:j-1
            ti = i*h
            J[i,j] = h*(ti*b)/2
        end

        for i in j:n
            ti = i*h
            J[i,j] = h*((1-ti)*a)/2
        end

        J[j,j] += 1
    end

    return J
end

function DInt(n::Int = 10)
    @assert n >= 1

    dint_n  = n
    dint_m  = n
    dint_x0 = [(ti = i/(n+1); ti*(ti-1)) for i in 1:n]
    dint_fx = [0.0]
    dint_x  = [[]]
    dint_data = nothing

    return MGH(
        "DInt-$n",
        dint_n,
        dint_m,
        dint_x0,
        dint_fx,
        dint_x,
        dint_data,
        dint_f!,
        dint_j!,
    )
end


#####
##### (30) Broyden tridiagonal function
#####
function tridiag_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n
    xi, xr = zero(eltype(x)), x[1]
    for i in 1:n-1
        xl, xi, xr = xi, xr, x[i+1]
        f[i] = (3 - 2*xi)*xi - xl - 2*xr + 1
    end
    xl, xi = xi, xr
    f[n] = (3 - 2*xi)*xi - xl + 1
    return f
end

function tridiag_j!(J, x, _)
    m, n = size(J)
    @assert m == n

    J[1,1] =  3 - 4*x[1]
    J[1,2] = -2
    for i in 3:n
        J[i,1] = 0
    end

    for j in 2:n-1
        xj = x[j]

        for i in 1:j-2
            J[i,j] = 0
        end

        J[j,j-1] = -1
        J[j,j]   =  3 - 4*xj
        J[j,j+1] = -2

        for i in j+2:n
            J[i,j] = 0
        end
    end

    for i in 1:n-2
        J[i,n] = 0
    end
    J[n,n-1] = -1
    J[n,n]   =  3 - 4*x[n]

    return J
end

function Tridiag(n::Int = 10)
    @assert n >= 1

    tridiag_n  = n # variable
    tridiag_m  = n # m = n
    tridiag_x0 = -ones(n)
    tridiag_fx = [0.0]
    tridiag_x  = [[]]
    tridiag_data = nothing

    return MGH(
        "Tridiag-$n",
        tridiag_n,
        tridiag_m,
        tridiag_x0,
        tridiag_fx,
        tridiag_x,
        tridiag_data,
        tridiag_f!,
        tridiag_j!,
    )
end


#####
##### (31) Broyden banded function
#####
function banded_f!(f, x, _)
    m, n = length(f), length(x)
    @assert m == n

    zeroT = zero(eltype(x))
    ml = 5
    mu = 1

    for i in 1:n
        i1 = max(1, i - ml)
        i2 = min(i + mu, n)
        c = zeroT
        for j in i1:i-1
            c += x[j]*(1 + x[j])
        end
        for j in i+1:i2
            c += x[j]*(1 + x[j])
        end
        xi = x[i]
        f[i] = xi*(2 + 5*xi*xi) + 1 - c
    end

    return f
end

function banded_j!(J, x, _)
    m, n = size(J)
    @assert m == n

    ml = 5
    mu = 1

    for j in 1:n
        xj = x[j]
        i1 = max(1, j - mu)
        i2 = min(j + ml, n)

        for i in 1:i1-1
            J[i,j] = 0
        end
        for i in i1:j-1
            J[i,j] = -(1 + 2*xj)
        end

        J[j,j] = 2 + 15*xj*xj

        for i in j+1:i2
            J[i,j] = -(1 + 2*xj)
        end
        for i in i2+1:n
            J[i,j] = 0
        end
    end

    return J
end

function Banded(n::Int = 10)
    @assert n >= 1

    banded_n  = n # variable
    banded_m  = n # m = n
    banded_x0 = -ones(n)
    banded_fx = [0.0]
    banded_x  = [[]]
    banded_data = nothing

    return MGH(
        "Banded-$n",
        banded_n,
        banded_m,
        banded_x0,
        banded_fx,
        banded_x,
        banded_data,
        banded_f!,
        banded_j!,
    )
end


#####
##### (32) Linear function - full rank
#####
function linear1_f!(f, x, _)
    m, n = length(f), length(x)
    s = -2/m * sum(x) - 1
    for i in 1:n
        f[i] = x[i] + s
    end
    for i in n+1:m
        f[i] = s
    end
    return f
end

function linear1_j!(J, x, _)
    m, n = size(J)
    c = -2/m
    for j in axes(J, 2)
        for i in axes(J, 1)
            J[i,j] = c
        end
        J[j,j] += 1
    end
    return J
end

function Linear1(n::Int = 5, m::Int = 10)
    @assert n >= 1
    @assert m >= n

    linear1_n  = n # variable
    linear1_m  = m # m >= n
    linear1_x0 = ones(n)
    linear1_fx = [m - n]
    linear1_x  = [-ones(n)]
    linear1_data = nothing

    return MGH(
        "Linear1-$n-$m",
        linear1_n,
        linear1_m,
        linear1_x0,
        linear1_fx,
        linear1_x,
        linear1_data,
        linear1_f!,
        linear1_j!,
    )
end


#####
##### (33) Linear function - rank 1
#####
function linear2_f!(f, x, _)
    s = zero(eltype(x))
    for j in eachindex(x)
        s += j*x[j]
    end
    for i in eachindex(f)
        f[i] = i*s - 1
    end
    return f
end

function linear2_j!(J, x, _)
    m, n = size(J)
    for j in axes(J, 2)
        for i in axes(J, 1)
            J[i,j] = i*j
        end
    end
    return J
end

function Linear2(n::Int = 5, m::Int = 10)
    @assert n >= 1
    @assert m >= n

    linear2_n  = n # variable
    linear2_m  = m # m >= n
    linear2_x0 = ones(n)
    linear2_fx = [m*(m - 1)/(2*(2*m + 1))]
    linear2_x  = [[]] # sum_i i*xi = 3 / (2*m + 1)
    linear2_data = nothing

    return MGH(
        "Linear2-$n-$m",
        linear2_n,
        linear2_m,
        linear2_x0,
        linear2_fx,
        linear2_x,
        linear2_data,
        linear2_f!,
        linear2_j!,
    )
end


#####
##### (34) Linear function - rank 1 with zero columns and rows
#####
function linear3_f!(f, x, _)
    m, n = length(f), length(x)
    s = zero(eltype(x))
    for j in 2:n-1
        s += j*x[j]
    end
    f[1] = -1
    for i in 2:m-1
        f[i] = (i-1)*s - 1
    end
    f[m] = -1
    return f
end

function linear3_j!(J, x, _)
    m, n = size(J)
    fill!(J, 0)
    for j in 2:n-1
        for i in 2:m-1
            J[i,j] = (i-1)*j
        end
    end
    return J
end

function Linear3(n::Int = 5, m::Int = 10)
    @assert n >= 1
    @assert m >= n

    linear3_n  = n # variable
    linear3_m  = m # m >= n
    linear3_x0 = ones(n)
    linear3_fx = [(m*m + 3*m - 6)/(2*(2*m - 3))]
    linear3_x  = [[]] # sum_i i*xi, i = 2:m-1 = 3 / (2*m - 3)
    linear3_data = nothing

    return MGH(
        "Linear3-$n-$m",
        linear3_n,
        linear3_m,
        linear3_x0,
        linear3_fx,
        linear3_x,
        linear3_data,
        linear3_f!,
        linear3_j!,
    )
end


#####
##### (35) Chebyquad function
#####
function chebyquad_f!(f, x, _)
    m, n = length(f), length(x)
    oneT = one(eltype(x))

    for i in 1:m
        f[i] = 0
    end

    for j in 1:n
        a = oneT
        b = 2*x[j] - 1
        c = 2*b
        for i in 1:m
            f[i] += b
            a, b = b, c*b - a
        end
    end

    dx = 1 / n
    ev = false

    for i in 1:m
        f[i] *= dx
        if ev
            f[i] += 1 / (i*i - 1)
        end
        ev ⊻= true
    end

    return f
end

function chebyquad_j!(J, x, _)
    m, n = size(J)
    oneT = one(eltype(x))
    zeroT = zero(eltype(x))
    dx = 1 / n
    for j in 1:n
        a = oneT
        b = 2*x[j] - 1
        c = 2*b
        d = zeroT
        e = 2*a
        for i in 1:m
            J[i,j] = dx*e
            d, e = e, 4*b + c*e - d
            a, b = b, c*b - a
        end
    end
    return J
end

function Chebyquad(n::Int = 5)
    @assert 1 <= n <= 10

    chebyquad_n = n
    chebyquad_m = n
    chebyquad_x0 = collect((1:n)./(n + 1))
    chebyquad_fx = [n == 10 ? 6.50395e-3 : n == 8 ? 3.51687e-3 : 0.0]
    chebyquad_x = [[]]
    chebyquad_data = nothing

    return MGH(
        "Chebyquad-$n",
        chebyquad_n,
        chebyquad_m,
        chebyquad_x0,
        chebyquad_fx,
        chebyquad_x,
        chebyquad_data,
        chebyquad_f!,
        chebyquad_j!,
    )
end


#####
##### Collections
#####

SystemsOfNonlinearEquations = [
    Rosenbrock(),
    Powell13(),
    Powell3(),
    Wood(),
    Helical(),
    Watson(6), Watson(9), Watson(12),
    Chebyquad(5), Chebyquad(6), Chebyquad(7), Chebyquad(8), Chebyquad(9), Chebyquad(10),
    Brown27(10), Brown27(30), Brown27(40),
    DBV(10),
    DInt(1), DInt(10),
    Trig(10),
    Vardim(10),
    Tridiag(10),
    Banded(10),
]

NonlinearLeastSquares = [
    Linear1(5, 10), Linear1(5, 50),
    Linear2(5, 10), Linear2(5, 50),
    Linear3(5, 10), Linear3(5, 50),
    Rosenbrock(),
    Helical(),
    Powell13(),
    Roth(),
    Bard(),
    Kowalik(),
    Meyer(),
    Watson(6), Watson(9), Watson(12),
    Box3d(10),
    Jennrich(),
    Dennis(),
    Chebyquad(5), Chebyquad(6), Chebyquad(7), Chebyquad(8), Chebyquad(9), Chebyquad(10),
    Brown27(10), Brown27(30), Brown27(40),
    Osborne1(),
    Osborne2(),
]

UnconstraindedMinimization = [
    Helical(),
    Biggs(13), Biggs(15),
    Gaussian(),
    Powell3(),
    Box3d(10),
    Vardim(10),
    Watson(6), Watson(9), Watson(12),
    Penalty1(4), Penalty1(10),
    Penalty2(4), Penalty2(10),
    Brown4(),
    Dennis(),
    Trig(10),
    RosenbrockN(24),
    Powell22(16),
    Beale(),
    Wood(),
    Chebyquad(5), Chebyquad(6), Chebyquad(7), Chebyquad(8), Chebyquad(9), Chebyquad(10),
]

TestProblems = [
    Rosenbrock(),
    Roth(),
    Powell3(),
    Brown4(),
    Beale(),
    Jennrich(),
    Helical(),
    Bard(),
    Gaussian(),
    Meyer(),
    Box3d(10),
    Powell13(),
    Wood(),
    Kowalik(),
    Dennis(),
    Osborne1(),
    Biggs(13), Biggs(15),
    Osborne2(),
    Watson(6), Watson(9), Watson(12),
    RosenbrockN(24),
    Powell22(16),
    Penalty1(4), Penalty1(10),
    Penalty2(4), Penalty2(10),
    Vardim(10),
    Trig(10),
    Brown27(10), Brown27(30), Brown27(40),
    DBV(10),
    DInt(1), DInt(10),
    Tridiag(10),
    Banded(10),
    Linear1(5, 10), Linear1(5, 50),
    Linear2(5, 10), Linear2(5, 50),
    Linear3(5, 10), Linear3(5, 50),
    Chebyquad(5), Chebyquad(6), Chebyquad(7), Chebyquad(8), Chebyquad(9), Chebyquad(10),
]

nothing
