"""Pruebas de estres: 10 casos distintos por bloque principal.

No sustituye a los tests finos de `test_debug.py`; esta suite busca que cada
tema aguante variedad de ecuaciones/datos sin excepciones ni resultados no
finitos.
"""

import contextlib
import io
import math

import numpy as np

import main
from utils import crear_funcion_segura

PASS = 0
FAIL = 0
ERRORS = []


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def ok(nombre, detalle=""):
    global PASS
    PASS += 1
    print(f"  OK    {nombre}")
    if detalle:
        print(f"        {detalle}")


def fail(nombre, detalle):
    global FAIL
    FAIL += 1
    ERRORS.append(f"{nombre}: {detalle}")
    print(f"  FAIL  {nombre}: {detalle}")


def assert_close(nombre, got, expected, tol):
    if got is None:
        fail(nombre, f"got None; expected {expected}")
        return
    if not math.isfinite(float(got)):
        fail(nombre, f"resultado no finito: {got}")
        return
    err = abs(float(got) - float(expected))
    if err <= tol:
        ok(nombre, f"{float(got):.10g} (err={err:.2e})")
    else:
        fail(nombre, f"got {got}, expected {expected}, err={err:.2e}, tol={tol}")


def assert_finite(nombre, value):
    arr = np.asarray(value, dtype=float)
    if arr.size and np.all(np.isfinite(arr)):
        ok(nombre)
    else:
        fail(nombre, f"valor no finito: {value}")


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def test_raices_10():
    section("RAICES: 10 ecuaciones con biseccion, regla falsa, secante y Newton")
    cases = [
        ("x^2 - 2", "2*x", "2", 1.0, 2.0, math.sqrt(2)),
        ("x^3 - x - 2", "3*x^2 - 1", "6*x", 1.0, 2.0, 1.5213797068045676),
        ("cos(x) - x", "-sin(x) - 1", "-cos(x)", 0.0, 1.0, 0.7390851332151607),
        ("exp(-x) - x", "-exp(-x) - 1", "exp(-x)", 0.0, 1.0, 0.5671432904097838),
        ("x^3 - 2", "3*x^2", "6*x", 1.0, 2.0, 2 ** (1 / 3)),
        ("sin(x) - 0.5", "cos(x)", "-sin(x)", 0.0, 1.0, math.asin(0.5)),
        ("x^2 - 5", "2*x", "2", 2.0, 3.0, math.sqrt(5)),
        ("log(x) - 1", "1/x", "-1/x^2", 2.0, 3.0, math.e),
        ("x^3 + 4*x^2 - 10", "3*x^2 + 8*x", "6*x + 8", 1.0, 2.0, 1.3652300134140969),
        ("x - 0.25", "1", "0", 0.0, 1.0, 0.25),
    ]
    methods = [
        ("biseccion", lambda f, df, d2, a, b: silent(main._biseccion_core, f, a, b, 1e-10, 250)[0], 1e-7),
        ("regla falsa", lambda f, df, d2, a, b: silent(main._regla_falsa_core, f, a, b, 1e-10, 250)[0], 1e-7),
        ("secante", lambda f, df, d2, a, b: silent(main._secante_core, f, a, b, 1e-10, 100)[0], 1e-7),
        ("Newton", lambda f, df, d2, a, b: silent(main._newton_core, f, df, (a + b) / 2, 1e-10, 100)[0], 1e-7),
    ]
    for expr, dexpr, d2expr, a, b, expected in cases:
        f = crear_funcion_segura(expr)
        df = crear_funcion_segura(dexpr)
        d2 = crear_funcion_segura(d2expr)
        for metodo, runner, tol in methods:
            try:
                got = runner(f, df, d2, a, b)
                assert_close(f"{metodo}: {expr}", got, expected, tol)
            except Exception as e:
                fail(f"{metodo}: {expr}", repr(e))


def test_interpolacion_10():
    section("INTERPOLACION: 10 funciones con Newton y Lagrange")
    cases = [
        ("x^2 + 1", lambda x: x**2 + 1),
        ("x^3 - x + 2", lambda x: x**3 - x + 2),
        ("sin(x)", np.sin),
        ("cos(x)", np.cos),
        ("exp(0.5*x)", lambda x: np.exp(0.5 * x)),
        ("log(x + 2)", lambda x: np.log(x + 2)),
        ("sqrt(x + 3)", lambda x: np.sqrt(x + 3)),
        ("1/(x + 2)", lambda x: 1 / (x + 2)),
        ("x^4 - 2*x^2 + 1", lambda x: x**4 - 2 * x**2 + 1),
        ("sin(x) + x", lambda x: np.sin(x) + x),
    ]
    x = np.linspace(-1.0, 1.0, 5)
    xp = 0.35
    for expr, fn in cases:
        y = fn(x)
        try:
            nval, _, tabla = silent(main.interpolacion_newton, x, y, xp)
            lval = silent(main.interpolacion_lagrange, x, y, xp)
            assert_close(f"Newton == Lagrange: {expr}", nval, lval, 1e-9)
            assert_finite(f"Tabla diferencias finita: {expr}", tabla)
        except Exception as e:
            fail(f"Interpolacion: {expr}", repr(e))


def test_regresion_10():
    section("REGRESION: 10 modelos/datasets")
    x = np.linspace(0.0, 4.0, 7)
    try:
        b0, b1, ec = silent(main.regresion_lineal, x, 2 * x + 1)
        assert_close("lineal y=2x+1 b0", b0, 1.0, 1e-10)
        assert_close("lineal y=2x+1 b1", b1, 2.0, 1e-10)
    except Exception as e:
        fail("lineal y=2x+1", repr(e))
    try:
        b0, b1, ec = silent(main.regresion_lineal, x, -3 * x + 5)
        assert_close("lineal y=-3x+5 b0", b0, 5.0, 1e-10)
        assert_close("lineal y=-3x+5 b1", b1, -3.0, 1e-10)
    except Exception as e:
        fail("lineal y=-3x+5", repr(e))

    poly_cases = [
        ("x^2+2x+1", x**2 + 2 * x + 1, 2, [1, 2, 1]),
        ("x^3-x+2", x**3 - x + 2, 3, [2, -1, 0, 1]),
    ]
    for name, y, grado, expected in poly_cases:
        try:
            coefs, ec = silent(main.regresion_polinomial, x, y, grado)
            for i, exp_i in enumerate(expected):
                assert_close(f"polinomial {name} b{i}", coefs[i], exp_i, 1e-8)
        except Exception as e:
            fail(f"polinomial {name}", repr(e))

    known_cases = [
        ("sin+cos", np.sin(x) + 2 * np.cos(x), [np.sin, np.cos], [1, 2]),
        ("exp+x", 3 * np.exp(0.2 * x) - 0.5 * x, [lambda z: np.exp(0.2 * z), lambda z: z], [3, -0.5]),
    ]
    for name, y, funcs, expected in known_cases:
        try:
            coefs, ec = silent(main.regresion_funcion_conocida, x, y, funcs)
            for i, exp_i in enumerate(expected):
                assert_close(f"funcion conocida {name} a{i}", coefs[i], exp_i, 1e-8)
        except Exception as e:
            fail(f"funcion conocida {name}", repr(e))

    exp_cases = [
        ("2e^(0.5x)", 2 * np.exp(0.5 * x), 0.5, 2.0),
        ("3e^(-0.2x)", 3 * np.exp(-0.2 * x), -0.2, 3.0),
    ]
    for name, y, a_exp, b_exp in exp_cases:
        try:
            a, b, ec = silent(main.regresion_exponencial, x, y)
            assert_close(f"exponencial {name} a", a, a_exp, 1e-8)
            assert_close(f"exponencial {name} b", b, b_exp, 1e-8)
        except Exception as e:
            fail(f"exponencial {name}", repr(e))

    xv = np.array([0, 1, 0, 1, 2, 2, 3], dtype=float)
    yv = np.array([0, 0, 1, 1, 1, 2, 3], dtype=float)
    multiple_cases = [
        ("1+2x+3y", 1 + 2 * xv + 3 * yv, [1, 2, 3]),
        ("-2+x-0.5y", -2 + xv - 0.5 * yv, [-2, 1, -0.5]),
    ]
    for name, z, expected in multiple_cases:
        try:
            coefs, ec = silent(main.regresion_multiple, xv, yv, z)
            for i, exp_i in enumerate(expected):
                assert_close(f"multiple {name} b{i}", coefs[i], exp_i, 1e-8)
        except Exception as e:
            fail(f"multiple {name}", repr(e))


def test_derivacion_integracion_10():
    section("DERIVACION E INTEGRACION: 10 funciones")
    cases = [
        ("x^2", lambda x: x**2, lambda x: 2*x, 0.0, 1.0, 1/3),
        ("x^3", lambda x: x**3, lambda x: 3*x**2, 0.0, 1.0, 1/4),
        ("sin(x)", np.sin, np.cos, 0.0, math.pi, 2.0),
        ("cos(x)", np.cos, lambda x: -np.sin(x), 0.0, math.pi / 2, 1.0),
        ("exp(x)", np.exp, np.exp, 0.0, 1.0, math.e - 1),
        ("1/(x+1)", lambda x: 1/(x+1), lambda x: -1/(x+1)**2, 0.0, 1.0, math.log(2)),
        ("sqrt(x+1)", lambda x: np.sqrt(x+1), lambda x: 1/(2*np.sqrt(x+1)), 0.0, 3.0, 14/3),
        ("log(x+1)", lambda x: np.log(x+1), lambda x: 1/(x+1), 0.0, 1.0, 2*math.log(2) - 1),
        ("x^4 - 2x + 1", lambda x: x**4 - 2*x + 1, lambda x: 4*x**3 - 2, 0.0, 1.0, 0.2),
        ("sin(x)+x", lambda x: np.sin(x)+x, lambda x: np.cos(x)+1, 0.0, 1.0, 1 - math.cos(1) + 0.5),
    ]
    h = 1e-3
    x0 = 0.5
    for name, fn, dfn, a, b, integral in cases:
        try:
            xs = np.array([x0 - h, x0, x0 + h])
            ys = fn(xs)
            d = silent(main._d1_hcte, xs, ys, "central", h)[0]["d"]
            assert_close(f"derivada central {name}", d, dfn(x0), 1e-4)
        except Exception as e:
            fail(f"derivada central {name}", repr(e))
        try:
            val = silent(main._integ_simpson13_comp_f, lambda z, _fn=fn: float(_fn(z)), a, b, 200)
            assert_close(f"integral Simpson 1/3 {name}", val, integral, 1e-6)
        except Exception as e:
            fail(f"integral Simpson 1/3 {name}", repr(e))


def test_edo_10():
    section("EDOS: 10 ecuaciones con Euler, RK2 y RK4")
    cases = [
        ("y", lambda x, y: y, 0, 1, 1, math.exp(1)),
        ("-y", lambda x, y: -y, 0, 1, 1, math.exp(-1)),
        ("x", lambda x, y: x, 0, 1, 1, 1.5),
        ("x + y", lambda x, y: x + y, 0, 1, 1, 2 * math.e - 2),
        ("y - x", lambda x, y: y - x, 0, 1, 1, 2.0),
        ("2xy", lambda x, y: 2 * x * y, 0, 1, 1, math.e),
        ("-2xy", lambda x, y: -2 * x * y, 0, 1, 1, math.exp(-1)),
        ("sin(x)", lambda x, y: math.sin(x), 0, 0, 1, 1 - math.cos(1)),
        ("cos(x)", lambda x, y: math.cos(x), 0, 0, 1, math.sin(1)),
        ("y(1-y)", lambda x, y: y * (1 - y), 0, 0.25, 1, 1 / (1 + 3 * math.exp(-1))),
    ]
    for name, f, x0, y0, xf, expected in cases:
        for metodo, fn in [
            ("Euler", main.metodo_euler),
            ("RK2 punto medio", main.metodo_rk2_pm),
            ("RK2 Heun", main.metodo_rk2_heun),
            ("RK4", main.metodo_rk4),
        ]:
            try:
                got = float(silent(fn, f, x0, y0, xf, 0.005))
                tol = 3e-2 if metodo == "Euler" else 5e-3 if metodo.startswith("RK2") else 1e-6
                assert_close(f"{metodo}: {name}", got, expected, tol)
            except Exception as e:
                fail(f"{metodo}: {name}", repr(e))


def solve_edp(exact, f_src, n=10, tol=1e-9):
    xa, xb, ya, yb = 0.0, 1.0, 0.0, 1.0
    xs = np.linspace(xa, xb, n + 1)
    ys = np.linspace(ya, yb, n + 1)
    hx = (xb - xa) / n
    hy = (yb - ya) / n
    u = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        u[i, 0] = exact(xs[i], 0.0)
        u[i, n] = exact(xs[i], 1.0)
    for j in range(n + 1):
        u[0, j] = exact(0.0, ys[j])
        u[n, j] = exact(1.0, ys[j])
    it, delta = silent(main._edp_solver, u, xs, ys, hx, hy, n, n, f_src, 1.4, tol, 50000)
    max_err = 0.0
    for i in range(n + 1):
        for j in range(n + 1):
            max_err = max(max_err, abs(u[i, j] - exact(xs[i], ys[j])))
    return max_err, it, delta


def test_edp_10():
    section("EDPS: 10 casos Laplace/Poisson")
    cases = [
        ("u=xy", lambda x, y: x*y, lambda x, y: 0.0, 1e-5),
        ("u=x^2-y^2", lambda x, y: x*x - y*y, lambda x, y: 0.0, 1e-5),
        ("u=x+y", lambda x, y: x + y, lambda x, y: 0.0, 1e-5),
        ("u=1+x-y", lambda x, y: 1 + x - y, lambda x, y: 0.0, 1e-5),
        ("u=x(1-x)/2+y(1-y)/2", lambda x, y: x*(1-x)/2 + y*(1-y)/2, lambda x, y: -2.0, 1e-5),
        ("u=x^2+y^2", lambda x, y: x*x + y*y, lambda x, y: 4.0, 1e-5),
        ("u=x^2", lambda x, y: x*x, lambda x, y: 2.0, 1e-5),
        ("u=y^2", lambda x, y: y*y, lambda x, y: 2.0, 1e-5),
        ("u=sin(pi*x)sin(pi*y)", lambda x, y: math.sin(math.pi*x)*math.sin(math.pi*y),
         lambda x, y: -2*math.pi**2*math.sin(math.pi*x)*math.sin(math.pi*y), 8e-3),
        ("u=x^3+y^3", lambda x, y: x**3 + y**3, lambda x, y: 6*x + 6*y, 1e-5),
    ]
    for name, exact, f_src, tol_err in cases:
        try:
            max_err, it, delta = solve_edp(exact, f_src, n=12)
            assert_close(f"EDP {name}", max_err, 0.0, tol_err)
        except Exception as e:
            fail(f"EDP {name}", repr(e))


for runner in [
    test_raices_10,
    test_interpolacion_10,
    test_regresion_10,
    test_derivacion_integracion_10,
    test_edo_10,
    test_edp_10,
]:
    runner()

total = PASS + FAIL
print(f"\nRESUMEN STRESS 10: {PASS} OK | {FAIL} FAIL | Total {total}")
if ERRORS:
    print("\nFallos:")
    for e in ERRORS:
        print(f"  {e}")
    raise SystemExit(1)
print("\nTodas las pruebas de estres pasaron.")
