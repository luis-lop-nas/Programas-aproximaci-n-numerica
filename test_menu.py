"""Smoke tests para menus interactivos."""

import builtins
import contextlib
import io

import main
from common import VolverAtras, _pedir_float
import tema_derivacion_integracion
import tema_edp
import tema_edo
import tema_interpolacion

PASS = 0
FAIL = 0
ERRORS = []


def run_with_input(fn, entradas):
    original_input = builtins.input
    it = iter(entradas)
    salida = io.StringIO()

    def fake_input(prompt=""):
        print(prompt, end="")
        return next(it)

    try:
        builtins.input = fake_input
        with contextlib.redirect_stdout(salida):
            fn()
    finally:
        builtins.input = original_input
    return salida.getvalue()


def check(nombre, fn):
    global PASS, FAIL
    try:
        detalle = fn()
    except Exception as e:
        FAIL += 1
        ERRORS.append(f"{nombre}: {e}")
        print(f"  FAIL  {nombre}: {e}")
        return
    PASS += 1
    print(f"  OK    {nombre}")
    if detalle:
        print(f"        {detalle}")


def test_main_sale():
    out = run_with_input(main.main, ["0"])
    assert "METODOS NUMERICOS" in out
    assert "Hasta luego" in out


def test_menu_global_vuelve():
    out = run_with_input(main.menu_graficar_global, ["0"])
    assert "GRAFICAR / COMPARAR" in out


def test_menu_global_opcion_invalida():
    out = run_with_input(main.menu_graficar_global, ["9", "0"])
    assert "Opcion invalida" in out


def test_main_entra_y_sale_de_graficar():
    out = run_with_input(main.main, ["6", "0", "0"])
    assert "GRAFICAR / COMPARAR" in out
    assert "Hasta luego" in out


def test_main_opcion_invalida():
    out = run_with_input(main.main, ["99", "0"])
    assert "Opcion invalida" in out


def test_menu_global_despacha_integracion():
    original = main.menu_graficar_integracion
    llamadas = []

    def fake_integracion():
        llamadas.append("integracion")

    try:
        main.menu_graficar_integracion = fake_integracion
        out = run_with_input(main.menu_graficar_global, ["3", "0"])
    finally:
        main.menu_graficar_integracion = original

    assert "Integracion" in out
    assert llamadas == ["integracion"]


def test_newton_pregunta_tabla_si():
    original_grafica = tema_interpolacion._preguntar_grafica
    try:
        tema_interpolacion._preguntar_grafica = lambda datos: None
        out = run_with_input(
            main.menu_interp_aprox,
            ["6", "0 1 2", "0 1 4", "", "1.5", "0"],
        )
    finally:
        tema_interpolacion._preguntar_grafica = original_grafica
    assert "Orden 1" in out
    assert "P(1.5)" in out


def test_newton_pregunta_tabla_no():
    original_grafica = tema_interpolacion._preguntar_grafica
    try:
        tema_interpolacion._preguntar_grafica = lambda datos: None
        out = run_with_input(
            main.menu_interp_aprox,
            ["6", "0 1 2", "0 1 4", "n", "1.5", "0"],
        )
    finally:
        tema_interpolacion._preguntar_grafica = original_grafica
    assert "Orden 1" not in out
    assert "P(1.5)" in out


def test_lagrange_bases_muestra_detalle():
    original_grafica = tema_interpolacion._preguntar_grafica
    try:
        tema_interpolacion._preguntar_grafica = lambda datos: None
        out = run_with_input(
            main.menu_interp_aprox,
            ["7", "0 1 2", "1 3 2", "", "1.5", "0"],
        )
    finally:
        tema_interpolacion._preguntar_grafica = original_grafica
    assert "Polinomios base de Lagrange" in out
    assert "L0(x)" in out
    assert "P(1.5)" in out


def test_integracion_compuesta_muestra_pesos():
    f = lambda x: x * x
    out = run_with_input(
        lambda: tema_derivacion_integracion._integ_trapecio_comp_f(f, 0.0, 1.0, 2),
        [],
    )
    assert "peso*f(xi)" in out
    assert "Suma ponderada" in out
    assert "I=" in out


def test_sistema_edo_muestra_k():
    f1 = lambda t, x, y: y
    f2 = lambda t, x, y: -x
    out = run_with_input(
        lambda: tema_edo._sistema2_run(
            f1, f2, "y", "-x", tema_edo._s2_rk4_paso, "RK4", 0.0, 1.0, 0.0, 0.1, 0.1
        ),
        [],
    )
    assert "k1x=" in out
    assert "k4y=" in out
    assert "x(t=0.1)" in out


def test_edp_solver_devuelve_historial():
    import numpy as np

    Nx = Ny = 2
    xs = np.linspace(0.0, 1.0, Nx + 1)
    ys = np.linspace(0.0, 1.0, Ny + 1)
    u = np.zeros((Nx + 1, Ny + 1))
    it, delta, hist = tema_edp._edp_solver(
        u, xs, ys, 0.5, 0.5, Nx, Ny, lambda x, y: 0.0, 1.0, 1e-6, 5, return_hist=True
    )
    assert it >= 1
    assert delta >= 0.0
    assert hist


def test_input_numerico_reintenta_mismo_prompt():
    out = run_with_input(lambda: print(f"valor={_pedir_float('x = '):g}"), ["abc", "2.5"])
    assert "Error: numero invalido" in out
    assert out.count("x = ") == 2
    assert "valor=2.5" in out


def test_input_esc_vuelve_atras():
    try:
        run_with_input(lambda: _pedir_float("x = "), ["esc"])
    except VolverAtras:
        return
    raise AssertionError("esc no lanzo VolverAtras")


for nombre, fn in [
    ("main sale con 0", test_main_sale),
    ("menu global vuelve con 0", test_menu_global_vuelve),
    ("menu global maneja opcion invalida", test_menu_global_opcion_invalida),
    ("main entra/sale de graficar", test_main_entra_y_sale_de_graficar),
    ("main maneja opcion invalida", test_main_opcion_invalida),
    ("menu global despacha integracion", test_menu_global_despacha_integracion),
    ("Newton muestra tabla si se pide", test_newton_pregunta_tabla_si),
    ("Newton oculta tabla si no se pide", test_newton_pregunta_tabla_no),
    ("Lagrange muestra polinomios base", test_lagrange_bases_muestra_detalle),
    ("Integracion compuesta muestra pesos", test_integracion_compuesta_muestra_pesos),
    ("Sistema EDO muestra k", test_sistema_edo_muestra_k),
    ("EDP solver devuelve historial", test_edp_solver_devuelve_historial),
    ("Input numerico reintenta", test_input_numerico_reintenta_mismo_prompt),
    ("Input esc vuelve atras", test_input_esc_vuelve_atras),
]:
    check(nombre, fn)

total = PASS + FAIL
print(f"\nRESUMEN MENU: {PASS} OK | {FAIL} FAIL | Total {total}")
if ERRORS:
    print("\nFallos:")
    for e in ERRORS:
        print(f"  {e}")
else:
    print("\nTodos los smoke tests de menu pasaron.")
