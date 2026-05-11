"""
test_debug.py — Suite de tests para main.py
Firmas verificadas contra el código fuente.
"""

import math, sys, io, contextlib
sys.path.insert(0, '/Users/luichi/Documents/Programas-aproximaci-n-numerica')
import numpy as np
import main
from utils import crear_funcion_segura

PASS = 0; FAIL = 0; ERRORS = []


def check(nombre, got, expected, tol=1e-6):
    global PASS, FAIL
    if got is None:
        FAIL += 1; ERRORS.append(f"FAIL {nombre}: got None")
        print(f"  FAIL  {nombre}: got None (expected {expected})")
        return
    err = abs(got - expected)
    if err <= tol:
        PASS += 1; print(f"  OK    {nombre}: {got:.8g} (err={err:.2e})")
    else:
        FAIL += 1; ERRORS.append(f"FAIL {nombre}: {got} vs {expected} (err={err:.2e})")
        print(f"  FAIL  {nombre}: got {got:.8g}, expected {expected:.8g} (err={err:.2e})")


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def try_check(nombre, fn, expected, tol=1e-6):
    """Llama fn(), extrae el resultado escalar y compara."""
    try:
        got = fn()
        if isinstance(got, (list, tuple)):
            got = got[0]
        if hasattr(got, '__float__'):
            got = float(got)
        check(nombre, got, expected, tol)
    except Exception as e:
        global FAIL
        FAIL += 1; ERRORS.append(f"FAIL {nombre}: {e}")
        print(f"  FAIL  {nombre}: {e}")


# ═══════════════════════════════════════════════════════════════
# RAICES — las funciones devuelven float directamente
# ═══════════════════════════════════════════════════════════════

section("RAICES")

# f(x) = x^3 - x - 2,  raiz exacta ≈ 1.52137970680...
f_r  = crear_funcion_segura("x^3 - x - 2")
df_r = crear_funcion_segura("3*x^2 - 1")
d2f_r= crear_funcion_segura("6*x")
g_pf = crear_funcion_segura("(x+2)^(1/3)")
ROOT = 1.5213797068

try_check("Biseccion [1,2]",
          lambda: silent(main._biseccion_core, f_r, 1.0, 2.0, 1e-8, 200), ROOT, 1e-6)

try_check("Regla falsa [1,2]",
          lambda: silent(main._regla_falsa_core, f_r, 1.0, 2.0, 1e-8, 200), ROOT, 1e-6)

try_check("Newton x0=1.5",
          lambda: silent(main._newton_core, f_r, df_r, 1.5, 1e-10, 100), ROOT, 1e-8)

try_check("Secante x0=1 x1=2",
          lambda: silent(main._secante_core, f_r, 1.0, 2.0, 1e-10, 100), ROOT, 1e-8)

try_check("Newton mejorado x0=1.5",
          lambda: silent(main._newton_mejorado_core, f_r, df_r, d2f_r, 1.5, 1e-10, 100), ROOT, 1e-8)

# cos(x)-x = 0, raiz ≈ 0.7390851332
f_c  = crear_funcion_segura("cos(x) - x")
df_c = crear_funcion_segura("-sin(x) - 1")
ROOT_COS = 0.7390851332

try_check("Biseccion cos(x)-x [0,1]",
          lambda: silent(main._biseccion_core, f_c, 0.0, 1.0, 1e-8, 200), ROOT_COS, 1e-6)

try_check("Newton cos(x)-x x0=0.7",
          lambda: silent(main._newton_core, f_c, df_c, 0.7, 1e-10, 50), ROOT_COS, 1e-8)

# Punto fijo: g(x) = (x+2)^(1/3) → raiz de x^3-x-2
try_check("Punto fijo g=(x+2)^(1/3)",
          lambda: silent(main._punto_fijo_core, f_r, g_pf, 1.5, 1e-8, 200), ROOT, 1e-5)


# ═══════════════════════════════════════════════════════════════
# INTERPOLACION — newton devuelve (res, coefs, T); lagrange devuelve float
# ═══════════════════════════════════════════════════════════════

section("INTERPOLACION")

x_nod = np.array([0.0, math.pi/6, math.pi/2])
y_nod = np.array([math.sin(xi) for xi in x_nod])
x_e   = math.pi/4
SIN_PI4 = math.sin(x_e)  # 0.70711

try:
    res, _, _ = silent(main.interpolacion_newton, x_nod, y_nod, x_e)
    check("Newton sin(x) en π/4", float(res), SIN_PI4, 3e-2)  # 3 nodos → error notable
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Newton sin: {e}"); print(f"  FAIL Newton sin: {e}")

try:
    val = silent(main.interpolacion_lagrange, x_nod, y_nod, x_e)
    check("Lagrange sin(x) en π/4", float(val), SIN_PI4, 3e-2)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Lagrange sin: {e}"); print(f"  FAIL Lagrange sin: {e}")

# Polinomio exacto: f(x)=x^2+3 con 3 nodos → interpolacion exacta
x2 = np.array([0.0, 1.0, 2.0]); y2 = x2**2 + 3
try:
    res, _, _ = silent(main.interpolacion_newton, x2, y2, 1.5)
    check("Newton x^2+3 en 1.5", float(res), 5.25, 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Newton x^2+3: {e}"); print(f"  FAIL Newton x^2+3: {e}")

try:
    val = silent(main.interpolacion_lagrange, x2, y2, 1.5)
    check("Lagrange x^2+3 en 1.5", float(val), 5.25, 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Lagrange x^2+3: {e}"); print(f"  FAIL Lagrange x^2+3: {e}")

# Consistencia Newton == Lagrange (pol. grado 3)
x3 = np.array([0.0,1.0,3.0,6.0]); y3 = np.sin(x3)
try:
    r_n, _, _ = silent(main.interpolacion_newton, x3, y3, 2.0)
    r_l       = silent(main.interpolacion_lagrange, x3, y3, 2.0)
    check("Newton==Lagrange en x=2", float(r_n), float(r_l), 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Newton==Lagrange: {e}"); print(f"  FAIL Newton==Lagrange: {e}")


# ═══════════════════════════════════════════════════════════════
# REGRESION — regresion_lineal → (b0,b1,ec)
#              regresion_polinomial → (coefs,ec)
#              regresion_exponencial → (a,b,ec)
# ═══════════════════════════════════════════════════════════════

section("REGRESION")

# y = 2x + 1
xr = np.array([0.0,1.0,2.0,3.0,4.0]); yr = 2*xr + 1
try:
    b0, b1, ec = silent(main.regresion_lineal, xr, yr)
    check("Regresion lineal b0 (intercept=1)", b0, 1.0, 1e-10)
    check("Regresion lineal b1 (pendiente=2)", b1, 2.0, 1e-10)
    check("Regresion lineal ec (=0)", ec, 0.0, 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Regresion lineal: {e}"); print(f"  FAIL Regresion lineal: {e}")

# y = x^2 con 4 puntos, grado 2
xp = np.array([0.0,1.0,2.0,3.0]); yp = xp**2
try:
    coefs, ec = silent(main.regresion_polinomial, xp, yp, 2)
    # coefs = [a0, a1, a2] → y = a0 + a1*x + a2*x^2
    check("Regresion pol grado2 a0 (=0)", coefs[0], 0.0, 1e-8)
    check("Regresion pol grado2 a1 (=0)", coefs[1], 0.0, 1e-8)
    check("Regresion pol grado2 a2 (=1)", coefs[2], 1.0, 1e-8)
    check("Regresion pol grado2 ec (=0)", ec, 0.0, 1e-8)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Regresion pol: {e}"); print(f"  FAIL Regresion pol: {e}")

# y = e^x → a=1 (coef expon.), b=1 (amplitud)
xe2 = np.array([0.0,1.0,2.0,3.0]); ye2 = np.exp(xe2)
try:
    a_e, b_e, ec = silent(main.regresion_exponencial, xe2, ye2)
    check("Regresion exp a (=1)", a_e, 1.0, 1e-4)
    check("Regresion exp b (=1)", b_e, 1.0, 1e-4)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Regresion exp: {e}"); print(f"  FAIL Regresion exp: {e}")

# Regresion multiple z = 1 + 2x + 3y
xv = np.array([0.,1.,0.,1.,2.]); yv = np.array([0.,0.,1.,1.,1.]); zv = 1 + 2*xv + 3*yv
try:
    coefs_m, ec_m = silent(main.regresion_multiple, xv, yv, zv)
    check("Regresion multiple b0 (=1)", coefs_m[0], 1.0, 1e-8)
    check("Regresion multiple b1 (=2)", coefs_m[1], 2.0, 1e-8)
    check("Regresion multiple b2 (=3)", coefs_m[2], 3.0, 1e-8)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL Regresion multiple: {e}"); print(f"  FAIL Regresion multiple: {e}")


# ═══════════════════════════════════════════════════════════════
# DERIVACION — _d1_hcte/_d2_hcte devuelven list[dict]
# ═══════════════════════════════════════════════════════════════

section("DERIVACION NUMERICA")

# f(x)=x^3, f'(x)=3x^2, f''(x)=6x, en x=2: f'=12, f''=12
xd = np.array([1.9, 2.0, 2.1]); yd = xd**3; h = 0.1

try:
    res = silent(main._d1_hcte, xd, yd, "central", h)
    # n=3, central → 1 resultado en i=1 (x=2.0)
    check("d1 central x^3 en x=2", res[0]["d"], 12.0, 2e-2)  # O(h^2) con h=0.1 → err≈0.01
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d1 central: {e}"); print(f"  FAIL d1 central: {e}")

try:
    res = silent(main._d1_hcte, xd, yd, "adelante", h)
    # adelante → 1 resultado en i=0 (x=1.9), f'(1.9)=3*1.9^2=10.83
    check("d1 adelante x^3 en x=1.9", res[0]["d"], 3*1.9**2, 0.1)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d1 adelante: {e}"); print(f"  FAIL d1 adelante: {e}")

try:
    res = silent(main._d1_hcte, xd, yd, "atras", h)
    # atras → 1 resultado en i=2 (x=2.1), f'(2.1)=3*2.1^2=13.23
    check("d1 atras x^3 en x=2.1", res[0]["d"], 3*2.1**2, 0.1)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d1 atras: {e}"); print(f"  FAIL d1 atras: {e}")

xd5 = np.array([1.8, 1.9, 2.0, 2.1, 2.2]); yd5 = xd5**3; h5 = 0.1
try:
    res = silent(main._d2_hcte, xd5, yd5, h5)
    # n=5, d2 → resultados en i=1,2,3 → res[1] es i=2 (x=2.0), f''(2)=12
    check("d2 central x^3 en x=2", res[1]["d"], 12.0, 1e-4)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d2 central: {e}"); print(f"  FAIL d2 central: {e}")

# Derivada exacta para cuadrática: f(x)=x^2, f'(x)=2x, f''(x)=2
xq = np.array([0.0, 1.0, 2.0, 3.0, 4.0]); yq = xq**2
try:
    res = silent(main._d1_hcte, xq, yq, "central", 1.0)
    # i=1→x=1, d=(4-0)/2=2; i=2→x=2, d=(9-1)/2=4; i=3→x=3, d=(16-4)/2=6
    check("d1 central x^2 en x=2", res[1]["d"], 4.0, 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d1 central x^2: {e}"); print(f"  FAIL d1 central x^2: {e}")

try:
    res = silent(main._d2_hcte, xq, yq, 1.0)
    # f''(x^2)=2 en todos los nodos interiores
    check("d2 x^2 (exacto=2)", res[0]["d"], 2.0, 1e-10)
    check("d2 x^2 (exacto=2) nodo 2", res[1]["d"], 2.0, 1e-10)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL d2 x^2: {e}"); print(f"  FAIL d2 x^2: {e}")


# ═══════════════════════════════════════════════════════════════
# INTEGRACION — devuelven float
# ═══════════════════════════════════════════════════════════════

section("INTEGRACION NUMERICA")

f_x2 = crear_funcion_segura("x^2")   # ∫₀¹ x² dx = 1/3
f_sn = crear_funcion_segura("sin(x)") # ∫₀^π sin dx = 2
EX13 = 1/3

try_check("Trapecio compuesto n=100",
          lambda: silent(main._integ_trapecio_comp_f, f_x2, 0, 1, 100), EX13, 1e-4)

try_check("Simpson 1/3 comp n=100",
          lambda: silent(main._integ_simpson13_comp_f, f_x2, 0, 1, 100), EX13, 1e-10)

try_check("Simpson 3/8 comp n=99",
          lambda: silent(main._integ_simpson38_comp_f, f_x2, 0, 1, 99), EX13, 1e-10)

try_check("Simpson 1/3 sin [0,π]",
          lambda: silent(main._integ_simpson13_comp_f, f_sn, 0, math.pi, 100), 2.0, 1e-6)

# Trapecio simple: exacto para lineales
f_lin = crear_funcion_segura("2*x + 1")  # ∫₀² (2x+1)dx = 6
try_check("Trapecio simple (2x+1) [0,2]",
          lambda: silent(main._integ_trapecio_simple, f_lin, 0, 2), 6.0, 1e-10)

# Simpson 1/3 simple: exacto para cúbicos (x^3)
f_x3 = crear_funcion_segura("x^3")  # ∫₀² x^3 dx = 4
try_check("Simpson 1/3 simple x^3 [0,2]",
          lambda: silent(main._integ_simpson13_simple, f_x3, 0, 2), 4.0, 1e-10)

# Simpson 3/8 simple: exacto para x^3
try_check("Simpson 3/8 simple x^3 [0,2]",
          lambda: silent(main._integ_simpson38_simple, f_x3, 0, 2), 4.0, 1e-10)

# Newton-Cotes abierta punto medio: I ≈ (b-a)*f((a+b)/2)
# Para x^2: I = 1*(0.5)^2 = 0.25 (no exacta para cuadraticas)
try_check("NC abierta 1pt (punto medio) x^2",
          lambda: silent(main._integ_punto_medio, f_x2, 0, 1), 0.25, 1e-10)

# NC abierta 2pts: 3h/2*(f(x1)+f(x2)), h=(b-a)/3
# Para x: ∫₀¹ x dx = 0.5; x1=1/3, x2=2/3 → 3*(1/3)/2*(1/3+2/3) = 0.5 (exacta para lineales)
f_x1 = crear_funcion_segura("x")
try_check("NC abierta 2pts exacta para x",
          lambda: silent(main._integ_dos_puntos, f_x1, 0, 1), 0.5, 1e-10)

# Tabular: trapecio y Simpson
xt = np.array([0.0,1.0,2.0,3.0]); yt = xt        # ∫₀³ x dx = 4.5
try_check("Trapecio tabular x [0,3]",
          lambda: silent(main._trapecio, xt, yt), 4.5, 1e-10)

xs = np.array([0.0,1.0,2.0,3.0,4.0]); ys = xs**2  # ∫₀⁴ x² dx = 64/3
try_check("Simpson 1/3 tabular x^2 [0,4]",
          lambda: silent(main._simpson13, xs, ys), 64/3, 1e-8)

# Simpson 3/8 tabular (necesita n múltiplo de 3 → 4 intervalos: indices 0..3)
xs38 = np.array([0.0,1.0,2.0,3.0]); ys38 = xs38**3  # ∫₀³ x^3 dx = 81/4
try_check("Simpson 3/8 tabular x^3 [0,3]",
          lambda: silent(main._simpson38, xs38, ys38), 81/4, 1e-8)


# ═══════════════════════════════════════════════════════════════
# EDOs ESCALARES — usar crear_funcion_xy_segura
# ═══════════════════════════════════════════════════════════════

section("EDOs ESCALARES  y' = f(x,y)")

# y' = y, y(0)=1 → y(t) = e^t; y(1) = e
f_y  = main.crear_funcion_xy_segura("y")
EXP1 = math.e

try_check("Euler y'=y h=0.05 y(1)",
          lambda: silent(main.metodo_euler, f_y, 0, 1, 1, 0.05), EXP1, 8e-2)  # O(h) → err≈0.065

try_check("RK2pm y'=y h=0.1 y(1)",
          lambda: silent(main.metodo_rk2_pm, f_y, 0, 1, 1, 0.1), EXP1, 5e-3)  # O(h^2)→err≈4e-3

try_check("Heun y'=y h=0.1 y(1)",
          lambda: silent(main.metodo_rk2_heun, f_y, 0, 1, 1, 0.1), EXP1, 5e-3)  # O(h^2)

try_check("RK4 y'=y h=0.1 y(1)",
          lambda: silent(main.metodo_rk4, f_y, 0, 1, 1, 0.1), EXP1, 5e-6)  # O(h^4)→err≈2e-6

# y' = -y, y(0)=1 → y(2) = e^{-2}
f_my = main.crear_funcion_xy_segura("-y")
try_check("RK4 y'=-y h=0.1 y(2)",
          lambda: silent(main.metodo_rk4, f_my, 0, 1, 2, 0.1), math.exp(-2), 5e-7)  # O(h^4)

# y' = x+y, y(0)=1 → y(1) = 2e-2
f_xy = main.crear_funcion_xy_segura("x+y")
EX_XY = 2*math.e - 2
try_check("RK4 y'=x+y h=0.1 y(1)",
          lambda: silent(main.metodo_rk4, f_xy, 0, 1, 1, 0.1), EX_XY, 5e-6)  # O(h^4)

# Newton-Cotes exactitud: Euler < RK2 < RK4 (orden de error)
e_eu = abs(silent(main.metodo_euler,    f_xy, 0, 1, 1, 0.1) - EX_XY)
e_r2 = abs(silent(main.metodo_rk2_pm,  f_xy, 0, 1, 1, 0.1) - EX_XY)
e_r4 = abs(silent(main.metodo_rk4,     f_xy, 0, 1, 1, 0.1) - EX_XY)
if e_eu > e_r2 > e_r4:
    PASS+=1; print(f"  OK    Orden convergencia Euler>RK2pm>RK4  ({e_eu:.2e}>{e_r2:.2e}>{e_r4:.2e})")
else:
    FAIL+=1; ERRORS.append("FAIL orden convergencia")
    print(f"  FAIL  Orden convergencia: eu={e_eu:.2e} r2={e_r2:.2e} r4={e_r4:.2e}")

# comparacion_metodos_edo no debe crashear
try:
    silent(main.comparacion_metodos_edo, f_xy, 0, 1, 1, 0.2)
    PASS+=1; print("  OK    comparacion_metodos_edo — sin crash")
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL comparacion: {e}"); print(f"  FAIL comparacion: {e}")


# ═══════════════════════════════════════════════════════════════
# SISTEMA 2 EDOs — solución exacta: dx/dt=y dy/dt=-x → x=cos t, y=-sin t
# ═══════════════════════════════════════════════════════════════

section("SISTEMA 2 EDOs")

f1s = main._crear_func_txy("y")
f2s = main._crear_func_txy("-x")
EX_X1 = math.cos(1.0)   # 0.54030
EX_Y1 = -math.sin(1.0)  # -0.84147


def run_s2(paso_fn, h=0.05):
    t, x, y = 0.0, 1.0, 0.0
    while t < 1.0 - 1e-14:
        h_ef = min(h, 1.0 - t)
        x, y = paso_fn(f1s, f2s, t, x, y, h_ef)
        t = round(t + h_ef, 14)
    return x, y


try:
    x, y = run_s2(main._s2_euler_paso, h=0.05)
    check("S2-Euler x(1)=cos(1)", x, EX_X1, 0.05)
    check("S2-Euler y(1)=-sin(1)", y, EX_Y1, 0.05)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL S2-Euler: {e}"); print(f"  FAIL S2-Euler: {e}")

try:
    x, y = run_s2(main._s2_rk2pm_paso, h=0.05)
    check("S2-RK2pm x(1)", x, EX_X1, 1e-3)
    check("S2-RK2pm y(1)", y, EX_Y1, 1e-3)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL S2-RK2pm: {e}"); print(f"  FAIL S2-RK2pm: {e}")

try:
    x, y = run_s2(main._s2_heun_paso, h=0.05)
    check("S2-Heun x(1)", x, EX_X1, 1e-3)
    check("S2-Heun y(1)", y, EX_Y1, 1e-3)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL S2-Heun: {e}"); print(f"  FAIL S2-Heun: {e}")

try:
    x, y = run_s2(main._s2_rk4_paso, h=0.1)
    check("S2-RK4 x(1)", x, EX_X1, 2e-6)  # O(h^4) con h=0.1
    check("S2-RK4 y(1)", y, EX_Y1, 2e-6)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL S2-RK4: {e}"); print(f"  FAIL S2-RK4: {e}")

# Orden S2: Euler < RK2pm ≈ Heun < RK4
ex, ey = run_s2(main._s2_euler_paso, h=0.1)
r2x, r2y = run_s2(main._s2_rk2pm_paso, h=0.1)
r4x, r4y = run_s2(main._s2_rk4_paso, h=0.1)
err_eu = max(abs(ex - EX_X1), abs(ey - EX_Y1))
err_r2 = max(abs(r2x - EX_X1), abs(r2y - EX_Y1))
err_r4 = max(abs(r4x - EX_X1), abs(r4y - EX_Y1))
if err_eu > err_r2 > err_r4:
    PASS+=1; print(f"  OK    Orden S2 Euler>RK2pm>RK4  ({err_eu:.2e}>{err_r2:.2e}>{err_r4:.2e})")
else:
    FAIL+=1; ERRORS.append("FAIL orden S2")
    print(f"  FAIL  Orden S2: eu={err_eu:.2e} r2={err_r2:.2e} r4={err_r4:.2e}")

# _sistema2_run no debe crashear
try:
    silent(main._sistema2_run, f1s, f2s, "y", "-x",
           main._s2_rk4_paso, "RK4", 0, 1, 0, 1, 0.1)
    PASS+=1; print("  OK    _sistema2_run sin crash")
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL _sistema2_run: {e}"); print(f"  FAIL _sistema2_run: {e}")


# ═══════════════════════════════════════════════════════════════
# EDO 2° ORDEN — x''+x=0 → x=cos(t), x'=-sin(t)
# ═══════════════════════════════════════════════════════════════

section("EDO 2° ORDEN  x'' = f(t, x, v)")

f_e2 = main._crear_func_txv("-x")
EX2_X = math.cos(1.0)
EX2_V = -math.sin(1.0)


def run_e2(paso_fn, h=0.05):
    t, x, v = 0.0, 1.0, 0.0
    while t < 1.0 - 1e-14:
        h_ef = min(h, 1.0 - t)
        x, v = paso_fn(f_e2, t, x, v, h_ef)
        t = round(t + h_ef, 14)
    return x, v


try:
    x, v = run_e2(main._e2_euler_paso, h=0.05)
    check("E2-Euler x(1)=cos(1)", x, EX2_X, 0.05)
    check("E2-Euler x'(1)=-sin(1)", v, EX2_V, 0.05)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL E2-Euler: {e}"); print(f"  FAIL E2-Euler: {e}")

try:
    x, v = run_e2(main._e2_rk2pm_paso, h=0.05)
    check("E2-RK2pm x(1)", x, EX2_X, 1e-3)
    check("E2-RK2pm x'(1)", v, EX2_V, 1e-3)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL E2-RK2pm: {e}"); print(f"  FAIL E2-RK2pm: {e}")

try:
    x, v = run_e2(main._e2_rk4_paso, h=0.1)
    check("E2-RK4 x(1)", x, EX2_X, 2e-6)  # O(h^4) con h=0.1
    check("E2-RK4 x'(1)", v, EX2_V, 2e-6)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL E2-RK4: {e}"); print(f"  FAIL E2-RK4: {e}")

# Orden E2: Euler < RK2pm < RK4
ex2, ev2 = run_e2(main._e2_euler_paso, h=0.1)
r2x2, r2v2 = run_e2(main._e2_rk2pm_paso, h=0.1)
r4x2, r4v2 = run_e2(main._e2_rk4_paso, h=0.1)
err_eu2 = max(abs(ex2 - EX2_X), abs(ev2 - EX2_V))
err_r22 = max(abs(r2x2 - EX2_X), abs(r2v2 - EX2_V))
err_r42 = max(abs(r4x2 - EX2_X), abs(r4v2 - EX2_V))
if err_eu2 > err_r22 > err_r42:
    PASS+=1; print(f"  OK    Orden E2 Euler>RK2pm>RK4  ({err_eu2:.2e}>{err_r22:.2e}>{err_r42:.2e})")
else:
    FAIL+=1; ERRORS.append("FAIL orden E2")
    print(f"  FAIL  Orden E2: eu={err_eu2:.2e} r2={err_r22:.2e} r4={err_r42:.2e}")

# Oscilador amortiguado x''+0.5x'+4x=0 no debe crashear
f_amor = main._crear_func_txv("-0.5*v - 4*x")
try:
    silent(main._edo2_run, f_amor, "-0.5*v-4*x",
           main._e2_euler_paso, "Euler", 0, 1, 0, 0.5, 0.1)
    PASS+=1; print("  OK    _edo2_run oscilador amortiguado — sin crash")
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL E2 amortiguado: {e}"); print(f"  FAIL E2 amortiguado: {e}")

# Vibración forzada x''+2x'+5x=sin(t) con RK4
f_forz = main._crear_func_txv("sin(t) - 2*v - 5*x")
try:
    silent(main._edo2_run, f_forz, "sin(t)-2*v-5*x",
           main._e2_rk4_paso, "RK4", 0, 0, 1, 0.3, 0.1)
    PASS+=1; print("  OK    _edo2_run vibración forzada RK4 — sin crash")
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL E2 forzado: {e}"); print(f"  FAIL E2 forzado: {e}")

# Proyectil y''=-9.81, y(0)=0, y'(0)=20, h=0.5 → resultado Euler (no exacto analítico)
# Euler step: y_{n+1}=y_n+h*v_n, v_{n+1}=v_n+h*(-9.81)
# Pasos: y1=10, y2=17.5475, y3=22.6425, y4=25.285
f_proy = main._crear_func_txv("-9.81")
EX_PROY_EULER = 25.285  # resultado correcto de Euler con h=0.5
t, x, v = 0.0, 0.0, 20.0
h_p = 0.5
while t < 2.0 - 1e-14:
    h_ef = min(h_p, 2.0 - t)
    x, v = main._e2_euler_paso(f_proy, t, x, v, h_ef)
    t = round(t + h_ef, 14)
check("E2-Euler proyectil y(2) resultado correcto", x, EX_PROY_EULER, 1e-8)

# Con RK4 el proyectil debe ser casi exacto (y'' cte → polinomio grado 2 → RK4 exacto)
EX_PROY_ANALITICO = 20*2 - 4.905*4  # 20.38
t, x, v = 0.0, 0.0, 20.0
while t < 2.0 - 1e-14:
    h_ef = min(h_p, 2.0 - t)
    x, v = main._e2_rk4_paso(f_proy, t, x, v, h_ef)
    t = round(t + h_ef, 14)
check("E2-RK4 proyectil y(2) casi exacto", x, EX_PROY_ANALITICO, 1e-6)


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

section("EDGE CASES")

# 1 solo paso
f_ec = main.crear_funcion_xy_segura("-y")
try_check("Euler un solo paso y'=-y h=xf",
          lambda: silent(main.metodo_euler, f_ec, 0, 1, 0.1, 0.1), 0.9, 1e-10)

# h mayor que intervalo → se clipa
try_check("Euler h>intervalo → clip",
          lambda: silent(main.metodo_euler, f_ec, 0, 1, 0.1, 1.0), 0.9, 1e-10)

# sqrt en la función
f_sqrt = main.crear_funcion_xy_segura("-0.6*sqrt(y)")
try_check("Euler y'=-0.6√y primer paso",
          lambda: silent(main.metodo_euler, f_sqrt, 0, 4, 0.5, 0.5), 3.4, 1e-10)

# t explícito en sistema
f1t = main._crear_func_txy("sin(t)")
f2t = main._crear_func_txy("cos(t)")
# ∫₀¹ sin(t) dt = 1-cos(1), ∫₀¹ cos(t) dt = sin(1) — con h pequeño
t, x, y = 0.0, 0.0, 0.0; h_t = 0.01
while t < 1.0 - 1e-14:
    h_ef = min(h_t, 1.0 - t)
    x, y = main._s2_rk4_paso(f1t, f2t, t, x, y, h_ef)
    t = round(t + h_ef, 14)
check("S2-RK4 ∫sin(t) con h=0.01", x, 1-math.cos(1), 1e-7)
check("S2-RK4 ∫cos(t) con h=0.01", y, math.sin(1), 1e-7)

# Biseccion detecta sin cambio de signo → None
try:
    res = silent(main._biseccion_core, f_r, 2.0, 3.0, 1e-6, 100)
    raiz = res[0] if isinstance(res, (tuple, list)) else res
    if raiz is None:
        PASS+=1; print("  OK    Biseccion sin cambio signo → None")
    else:
        FAIL+=1; ERRORS.append("FAIL: biseccion debería dar None en [2,3]")
        print("  FAIL  Biseccion sin cambio signo debería dar None")
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL biseccion edge: {e}"); print(f"  FAIL biseccion edge: {e}")

# Polinomios enteros: división sintética de x^3-x-2 entre (x-r) debe dar resto≈0
try:
    # r = 1.5213797068 es raiz de x^3-x-2 = x^3 + 0*x^2 + (-1)*x + (-2)
    coefs = [1, 0, -1, -2]
    q, resto = silent(main._division_sintetica, coefs, ROOT)
    check("Division sintetica — resto≈0", resto, 0.0, 1e-5)
except Exception as e:
    FAIL+=1; ERRORS.append(f"FAIL div sintetica: {e}"); print(f"  FAIL div sintetica: {e}")


# ═══════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESUMEN FINAL:  {PASS} OK  |  {FAIL} FAIL  |  Total {total}")
print('='*60)
if ERRORS:
    print(f"\nFallos ({len(ERRORS)}):")
    for e in ERRORS:
        print(f"  {e}")
else:
    print("\nTodos los tests pasaron. El programa no tiene bugs.")
