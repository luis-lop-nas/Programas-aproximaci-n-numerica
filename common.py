"""
common.py — Utilidades compartidas para los modulos de metodos numericos.
"""

import math
import os
import tempfile
import numpy as np
from utils import (
    crear_evaluador_seguro, crear_funcion_segura, derivada_simbolica, derivadas_simbolicas,
    buscar_cambios_de_signo, refinar_cambio, sugerir_intervalos, AYUDA_FUNCIONES,
)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))
try:
    import graficar as _gr
    _MPLOK = True
except ImportError:
    _MPLOK = False


# ─── Entorno seguro para f(x,y) en EDOs ──────────────────────────────────────
_SAFE_XY = {
    "__builtins__": {},
    "math": math, "abs": abs, "pow": pow,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "log2": math.log2, "log10": math.log10,
    "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
}


def crear_funcion_xy_segura(f_str):
    evaluar = crear_evaluador_seguro(f_str, ("x", "y"))
    def f(x, y):
        return evaluar(x=x, y=y)
    return f


# ─── Helpers comunes ──────────────────────────────────────────────────────────

class VolverAtras(BaseException):
    """Senyal interna para volver un nivel en los menus interactivos."""


_ESC_TOKENS = {"\x1b", "esc", "escape", ":q", "volver", "atras", "atrás"}


def _es_escape(valor):
    return valor.strip().lower() in _ESC_TOKENS


def _input(prompt=""):
    valor = input(prompt)
    if _es_escape(valor):
        raise VolverAtras()
    return valor


def _avisar_esc():
    print("  (Escribe 'esc' para volver atras)")


def _pedir_texto(prompt, *, permitir_vacio=False, normalizar=None):
    while True:
        valor = _input(prompt).strip()
        if not valor and not permitir_vacio:
            print("Error: valor vacio.")
            continue
        return normalizar(valor) if normalizar else valor


def _pedir_float(prompt, *, condicion=None, error="numero invalido", permitir_vacio=False, default=None):
    while True:
        raw = _input(prompt).strip()
        if permitir_vacio and raw == "":
            return default
        try:
            valor = float(raw)
        except Exception:
            print(f"Error: {error}.")
            continue
        if not math.isfinite(valor):
            print("Error: NaN/Inf no es valido.")
            continue
        if condicion and not condicion(valor):
            print(f"Error: {error}.")
            continue
        return valor


def _pedir_int(prompt, *, condicion=None, error="entero invalido", permitir_vacio=False, default=None):
    while True:
        raw = _input(prompt).strip()
        if permitir_vacio and raw == "":
            return default
        try:
            valor = int(raw)
        except Exception:
            print(f"Error: {error}.")
            continue
        if condicion and not condicion(valor):
            print(f"Error: {error}.")
            continue
        return valor


def _pedir_float_array(prompt, *, min_len=1):
    while True:
        raw = _input(prompt).strip()
        try:
            arr = np.array([float(v) for v in raw.split()], dtype=float)
        except Exception:
            print("Error: solo numeros separados por espacios.")
            continue
        if arr.size < min_len:
            print(f"Error: minimo {min_len} valor(es).")
            continue
        if not np.all(np.isfinite(arr)):
            print("Error: NaN/Inf en los datos.")
            continue
        return arr


def _en(xn, xa):
    return abs(xn - xa) / abs(xn) if abs(xn) > 1e-15 else abs(xn - xa)


def _pedir_funcion():
    print(AYUDA_FUNCIONES)
    print("\nEjemplos:  x^3 - 2*x - 5  |  sin(x) - x/2  |  exp(x) - 3*x\n")
    _avisar_esc()
    while True:
        f_str = _pedir_texto("f(x) = ")
        try:
            f = crear_funcion_segura(f_str); f(1)
        except Exception as e:
            print(f"Error: {e}")
            continue
        return f, f_str


def _pedir_tol_iter(def_tol=1e-3, def_iter=100):
    tol = _pedir_float(
        f"Tolerancia (Enter={def_tol}): ",
        condicion=lambda v: v > 0,
        error="la tolerancia debe ser > 0",
        permitir_vacio=True,
        default=def_tol,
    )
    maxit = _pedir_int(
        f"Max iteraciones (Enter={def_iter}): ",
        condicion=lambda v: v > 0,
        error="las iteraciones deben ser > 0",
        permitir_vacio=True,
        default=def_iter,
    )
    return tol, maxit


def _pedir_intervalo():
    while True:
        a = _pedir_float("a = ")
        b = _pedir_float("b = ")
        if a >= b:
            print("Error: a debe ser menor que b")
            continue
        return a, b


def _preguntar_grafica(datos):
    """Pregunta si graficar y muestra la grafica en ventana nueva segun el tipo."""
    if not _MPLOK:
        return
    try:
        resp = _input("\n¿Graficar resultado? (s/n): ").strip().lower()
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    if resp not in ('s', 'si', 'sí', 'y', 'yes', '1'):
        return
    tipo = datos.get('tipo', '')
    if tipo == 'raiz':
        _gr.raiz_fx(datos['f'], datos['a'], datos['b'],
                    datos.get('raiz'), datos.get('f_str', 'f'), datos.get('metodo', ''))
    elif tipo == 'raiz_conv':
        _gr.raiz_convergencia({datos.get('metodo', 'metodo'): datos.get('hist_en', [])})
    elif tipo == 'edo':
        _gr.edo_escalar([{'xs': datos['xs'], 'ys': datos['ys'],
                          'label': datos.get('metodo', '')}],
                        titulo=datos.get('titulo', 'EDO'))
    elif tipo == 'edo_sistema':
        _gr.edo_sistema([{'t': datos['t'], 'x': datos['x'], 'y': datos['y'],
                          'label': datos.get('metodo', '')}],
                        eje='ambas', titulo=datos.get('titulo', 'Sistema 2 EDOs'))
    elif tipo == 'edo_2orden':
        _gr.edo_2orden([{'t': datos['t'], 'x': datos['x'], 'v': datos['v'],
                         'label': datos.get('metodo', '')}],
                       eje='ambas', titulo=datos.get('titulo', 'EDO 2° orden'))
    elif tipo == 'integracion':
        _gr.integracion(datos['f'], datos['a'], datos['b'],
                        datos.get('n', 10), datos.get('metodo', ''), datos['resultado'])
    elif tipo == 'interpolacion':
        x_d = np.array(datos['x_datos']); y_d = np.array(datos['y_datos'])
        x_fine = np.linspace(float(x_d.min()), float(x_d.max()), 400)
        try:
            y_fine = np.array([datos['f_ajuste'](xi) for xi in x_fine], dtype=float)
            _gr.interpolacion(x_d, y_d,
                              [{'xs': x_fine, 'ys': y_fine, 'label': datos.get('label', '')}],
                              titulo=datos.get('titulo', 'Ajuste'))
        except Exception as e:
            print(f"  Error al graficar: {e}")
    elif tipo == 'edp':
        _gr.edp(datos['u'], datos['xs'], datos['ys'], datos.get('titulo', 'EDP'))




__all__ = [name for name in globals() if not name.startswith("__")]
