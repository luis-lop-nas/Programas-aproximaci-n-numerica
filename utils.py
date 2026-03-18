"""
utils.py — Utilidades compartidas por todos los metodos numericos.

Importar en cada main.py con:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import crear_funcion_segura, derivada_simbolica, derivadas_simbolicas, \
                      buscar_cambios_de_signo, refinar_cambio
"""

import math
import re


# ─────────────────────────────────────────────────────────────
# Evaluacion segura de funciones introducidas por el usuario
# ─────────────────────────────────────────────────────────────

# Entorno de evaluacion controlado: solo se permiten estas funciones/constantes.
# __builtins__ vacio bloquea todas las funciones peligrosas de Python (open, exec, etc.)
_ALLOWED_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "abs":   abs,
    "pow":   pow,
    # Trigonometricas basicas
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    # Trigonometricas inversas
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "atan2": math.atan2,
    # Hiperbolicas
    "sinh":  math.sinh,
    "cosh":  math.cosh,
    "tanh":  math.tanh,
    # Exponencial y logaritmos
    "exp":   math.exp,
    "log":   math.log,    # log(x) = ln(x) ; log(x, base) tambien funciona
    "log2":  math.log2,
    "log10": math.log10,
    "sqrt":  math.sqrt,
    # Constantes
    "pi":    math.pi,
    "e":     math.e,
}

# Texto de ayuda con todas las funciones disponibles (se muestra al usuario)
AYUDA_FUNCIONES = (
    "Funciones disponibles:\n"
    "  Basicas    : sin, cos, tan, exp, log (=ln), sqrt, abs\n"
    "  Inversas   : asin, acos, atan\n"
    "  Hiperbolicas: sinh, cosh, tanh\n"
    "  Logaritmos : log2, log10\n"
    "Constantes   : pi, e\n"
    "Potencias    : usa ^ o **"
)


def crear_funcion_segura(f_str):
    """
    Convierte el string de la funcion introducida por el usuario en un callable f(x).
    Sustituye ^ por ** y ln( por log( para compatibilidad.
    Evalua en un entorno restringido que solo permite las funciones matematicas de _ALLOWED_GLOBALS.
    """
    expr = (
        f_str.strip()
        .replace("^", "**")
        .replace("ln(", "log(")      # ln -> log (ambos son logaritmo natural aqui)
        .replace("math.log(", "log(")  # por si el usuario escribe math.log directamente
    )

    def f(x):
        return eval(expr, _ALLOWED_GLOBALS, {"x": x})

    return f


# ─────────────────────────────────────────────────────────────
# Diferenciacion simbolica con sympy
# ─────────────────────────────────────────────────────────────

def _preparar_expr_sympy(expr_str):
    """Adapta el string de la funcion al formato que entiende sympy."""
    s = expr_str.strip().replace("^", "**")
    s = s.replace("ln(", "log(")        # sympy usa log() para logaritmo natural
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bmath\.", "", s)       # elimina prefijos math. sobrantes
    return s


def derivada_simbolica(expr_str):
    """
    Calcula f'(x) de forma simbolica usando sympy.
    Devuelve (df_func, df_str): la funcion numerica y su representacion en texto.
    """
    try:
        from sympy import symbols, diff, lambdify
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
    except Exception as e:
        raise RuntimeError("Falta sympy. Instalalo con: pip install sympy") from e

    x = symbols("x")
    s = _preparar_expr_sympy(expr_str)
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    df_expr = diff(expr, x)
    df_func = lambdify(x, df_expr, modules="math")
    return df_func, str(df_expr)


def derivadas_simbolicas(expr_str):
    """
    Calcula f'(x) y f''(x) de forma simbolica usando sympy.
    Devuelve (df_func, df_str, d2f_func, d2f_str).
    """
    try:
        from sympy import symbols, diff, lambdify
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
    except Exception as e:
        raise RuntimeError("Falta sympy. Instalalo con: pip install sympy") from e

    x = symbols("x")
    s = _preparar_expr_sympy(expr_str)
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    df_expr  = diff(expr, x)
    d2f_expr = diff(df_expr, x)

    df_func  = lambdify(x, df_expr,  modules="math")
    d2f_func = lambdify(x, d2f_expr, modules="math")
    return df_func, str(df_expr), d2f_func, str(d2f_expr)


# ─────────────────────────────────────────────────────────────
# Deteccion de cambios de signo (Teorema de Bolzano)
# ─────────────────────────────────────────────────────────────

def buscar_cambios_de_signo(f, a, b, n_subdivisiones=1000):
    """
    Divide [a, b] en n_subdivisiones subintervalos iguales y detecta todos
    los subintervalos donde f cambia de signo (Bolzano).
    Devuelve lista de tuplas (x_izq, x_der, f_izq, f_der).
    """
    paso   = (b - a) / n_subdivisiones
    cambios = []

    x_izq = a
    try:
        f_izq = f(x_izq)
    except Exception:
        f_izq = None

    for i in range(n_subdivisiones):
        x_der = a + (i + 1) * paso
        try:
            f_der = f(x_der)
        except Exception:
            f_izq = None
            x_izq = x_der
            continue

        if f_izq is None or not math.isfinite(f_izq) or not math.isfinite(f_der):
            f_izq = f_der
            x_izq = x_der
            continue

        if f_izq * f_der < 0:
            cambios.append((x_izq, x_der, f_izq, f_der))

        x_izq = x_der
        f_izq = f_der

    return cambios


def refinar_cambio(f, xi, xd, iteraciones=50):
    """
    Localiza con precision la raiz dentro de [xi, xd] usando biseccion interna.
    """
    a, b = xi, xd
    for _ in range(iteraciones):
        m = (a + b) / 2
        try:
            fm = f(m)
        except Exception:
            break
        if not math.isfinite(fm):
            break
        if f(a) * fm < 0:
            b = m
        else:
            a = m
    return (a + b) / 2


def sugerir_intervalos(f, a, b, n=1000):
    """
    Si no hay cambio de signo global en [a, b], busca subintervalos validos
    y los muestra al usuario. Devuelve la lista de cambios encontrados.
    """
    cambios = buscar_cambios_de_signo(f, a, b, n)
    if not cambios:
        print("  No se encontro ningun cambio de signo interior.")
        print("  Prueba con un intervalo diferente o comprueba la funcion.")
    else:
        print(f"  Se encontraron {len(cambios)} subintervalo(s) con cambio de signo:")
        for i, (xi, xd, fi, fd) in enumerate(cambios, 1):
            raiz = refinar_cambio(f, xi, xd)
            print(f"    {i}. [{xi:.6f}, {xd:.6f}]  ->  raiz ≈ {raiz:.8f}")
        print("  Usa uno de esos subintervalos como [a, b].")
    return cambios
