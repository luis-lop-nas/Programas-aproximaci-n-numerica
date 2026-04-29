"""
main.py — Menu unificado de metodos numericos.
Temas: Raices | Interpolacion y Aproximacion | Derivacion e Integracion | EDOs
Ejecutar:  python main.py
"""

import math
import numpy as np
from utils import (
    crear_funcion_segura, derivada_simbolica, derivadas_simbolicas,
    buscar_cambios_de_signo, refinar_cambio, sugerir_intervalos, AYUDA_FUNCIONES,
)


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
    expr = f_str.strip().replace("^", "**").replace("ln(", "log(").replace("math.log(", "log(")
    def f(x, y):
        return eval(expr, _SAFE_XY, {"x": x, "y": y})
    return f


# ─── Helpers comunes ──────────────────────────────────────────────────────────

def _en(xn, xa):
    return abs(xn - xa) / abs(xn) if abs(xn) > 1e-15 else abs(xn - xa)


def _pedir_funcion():
    print(AYUDA_FUNCIONES)
    print("\nEjemplos:  x^3 - 2*x - 5  |  sin(x) - x/2  |  exp(x) - 3*x\n")
    f_str = input("f(x) = ").strip()
    try:
        f = crear_funcion_segura(f_str); f(1)
    except Exception as e:
        print(f"Error: {e}"); return None, None
    return f, f_str


def _pedir_tol_iter(def_tol=1e-3, def_iter=100):
    try:
        t = input(f"Tolerancia (Enter={def_tol}): ").strip()
        tol = float(t) if t else def_tol
        if tol <= 0: tol = def_tol
    except Exception:
        tol = def_tol
    try:
        it = input(f"Max iteraciones (Enter={def_iter}): ").strip()
        maxit = int(it) if it else def_iter
        if maxit <= 0: maxit = def_iter
    except Exception:
        maxit = def_iter
    return tol, maxit


def _pedir_intervalo():
    try:
        a = float(input("a = "))
        b = float(input("b = "))
        if a >= b:
            print("Error: a debe ser menor que b"); return None, None
    except Exception:
        print("Error: ingresa numeros validos"); return None, None
    return a, b


# ═══════════════════════════════════════════════════════════════════════════════
# RAICES DE ECUACIONES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Bolzano ──────────────────────────────────────────────────────────────────

def menu_bolzano():
    print("\n=== TEOREMA DE BOLZANO ===\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    try:
        n_str = input("Subdivisiones (Enter=1000): ").strip()
        n = int(n_str) if n_str else 1000
        if n <= 0: n = 1000
    except Exception:
        n = 1000
    try:
        fa, fb = f(a), f(b)
    except Exception as e:
        print(f"Error evaluando f: {e}"); return
    print(f"\nf({a}) = {fa:.6f}   f({b}) = {fb:.6f}")
    if math.isfinite(fa) and math.isfinite(fb):
        if fa * fb < 0:
            print("Bolzano en [a,b]: SI hay cambio de signo -> existe al menos una raiz.")
        elif fa * fb > 0:
            print("Bolzano en [a,b]: NO hay cambio de signo en los extremos.")
        else:
            if fa == 0: print(f"a={a} es raiz exacta.")
            if fb == 0: print(f"b={b} es raiz exacta.")
    cambios = buscar_cambios_de_signo(f, a, b, n)
    print(f"\n{'─'*70}")
    if not cambios:
        print("No se detecto ningun cambio de signo interior.")
        print("Prueba con otro intervalo o mas subdivisiones.")
        return
    print(f"Se encontraron {len(cambios)} cambio(s) de signo:\n")
    print(f"  {'#':<5} {'Subintervalo':<30} {'Raiz aproximada'}")
    print(f"  {'─'*55}")
    for i, (xi, xd, fi, fd) in enumerate(cambios, 1):
        raiz = refinar_cambio(f, xi, xd)
        print(f"  {i:<5} [{xi:.6f}, {xd:.6f}]   x ≈ {raiz:.10f}")


# ─── Biseccion ────────────────────────────────────────────────────────────────

def _biseccion_core(funcion, a, b, tol, maxit):
    fa, fb = funcion(a), funcion(b)
    if fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(funcion, a, b); return None
    print(f"\n{'Iter':<6} {'a':<18} {'b':<18} {'c':<18} {'f(c)':<18} {'EN':<15}")
    print("-" * 95)
    c_ant = None
    for it in range(maxit):
        c = (a + b) / 2; fc = funcion(c)
        EN = _en(c, c_ant) if c_ant is not None else float('inf')
        if c_ant is None:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {EN:<15.8e}")
        if c_ant is not None and EN < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return c
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        c_ant = c
    c = (a + b) / 2; fc = funcion(c)
    print(f"\nMax iter. Raiz aprox: x = {c:.10f}   f(x) = {fc:.10e}")
    return c


def menu_biseccion():
    print("\n=== METODO DE BISECCION ===\n")
    print("c = (a + b) / 2\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    tol, maxit = _pedir_tol_iter(def_iter=200)
    print(f"\nResolviendo f(x) = {f_str}  en  [{a}, {b}]")
    _biseccion_core(f, a, b, tol, maxit)


# ─── Regla Falsa ──────────────────────────────────────────────────────────────

def _regla_falsa_core(funcion, a, b, tol, maxit):
    fa, fb = funcion(a), funcion(b)
    if fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(funcion, a, b); return None
    print(f"\n{'Iter':<6} {'a':<18} {'b':<18} {'c':<18} {'f(c)':<18} {'EN':<15}")
    print("-" * 95)
    c_ant = c = None
    for it in range(maxit):
        d = fb - fa
        if abs(d) < 1e-15:
            print("Error: f(b)-f(a)~0"); return None
        c = b - fb * (b - a) / d; fc = funcion(c)
        EN = _en(c, c_ant) if c_ant is not None else float('inf')
        if c_ant is None:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {EN:<15.8e}")
        if c_ant is not None and EN < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return c
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        c_ant = c
    print(f"\nMax iter. Raiz aprox: x = {c:.10f}   f(x) = {funcion(c):.10e}")
    return c


def menu_regla_falsa():
    print("\n=== METODO DE REGLA FALSA ===\n")
    print("c = b - f(b)*(b-a) / (f(b)-f(a))\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x) = {f_str}  en  [{a}, {b}]")
    _regla_falsa_core(f, a, b, tol, maxit)


# ─── Punto Fijo ───────────────────────────────────────────────────────────────

def _punto_fijo_core(f, g, x0, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n':<22} {'x_n+1':<22} {'f(x_n+1)':<22} {'EN':<15}")
    print("-" * 92)
    x = x0
    for it in range(maxit):
        try:
            xn = g(x)
        except Exception as e:
            print(f"\nError g(x): {e}"); return None
        if not math.isfinite(xn):
            print("Error: g(x) diverge"); return None
        try:
            fn = f(xn)
        except Exception:
            fn = float('nan')
        EN = _en(xn, x)
        if it == 0:
            print(f"{it:<6} {x:<22.10f} {xn:<22.10f} {fn:<22.10e} {'---':<15}")
        else:
            print(f"{it:<6} {x:<22.10f} {xn:<22.10f} {fn:<22.10e} {EN:<15.8e}")
        if it > 0 and EN < tol:
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}")
    try:
        print(f"f({x:.10f}) = {f(x):.10e}")
    except Exception:
        pass
    return x


def menu_punto_fijo():
    print("\n=== METODO DE PUNTO FIJO ===\n")
    print("x_{n+1} = g(x_n)  donde  f(x)=0  se reescribe como  x=g(x)\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    g_str = input("g(x) = ").strip()
    try:
        g = crear_funcion_segura(g_str); g(1)
    except Exception as e:
        print(f"Error g(x): {e}"); return
    try:
        x0 = float(input("\nx0 = "))
    except Exception:
        print("Error: numero invalido"); return
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  g(x)={g_str}  x0={x0}")
    _punto_fijo_core(f, g, x0, tol, maxit)


# ─── Secante ──────────────────────────────────────────────────────────────────

def _secante_core(f, x0, x1, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n-1':<22} {'x_n':<22} {'x_n+1':<22} {'f(x_n+1)':<18} {'EN':<15}")
    print("-" * 110)
    xa, xb = x0, x1
    for it in range(maxit):
        try:
            fa, fb = f(xa), f(xb)
        except Exception as e:
            print(f"\nError: {e}"); return None
        if not (math.isfinite(fa) and math.isfinite(fb)):
            print("Error: f(x) no finito"); return None
        d = fb - fa
        if abs(d) < 1e-15:
            print(f"Error: f(x_n)-f(x_{{n-1}})~0 en iter {it}"); return None
        xn = xb - fb * (xb - xa) / d
        if not math.isfinite(xn):
            print("Error: x diverge"); return None
        fn = f(xn); EN = _en(xn, xb)
        if it == 0:
            print(f"{it:<6} {xa:<22.10f} {xb:<22.10f} {xn:<22.10f} {fn:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {xa:<22.10f} {xb:<22.10f} {xn:<22.10f} {fn:<18.10e} {EN:<15.8e}")
        if it > 0 and EN < tol:
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn
        xa, xb = xb, xn
    print(f"\nMax iter. Raiz aprox: x = {xb:.10f}   f(x) = {f(xb):.10e}")
    return xb


def menu_secante():
    print("\n=== METODO DE LA SECANTE ===\n")
    print("x_{n+1} = x_n - f(x_n)*(x_n-x_{n-1}) / (f(x_n)-f(x_{n-1}))\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    try:
        x0 = float(input("x0 = "))
        x1 = float(input("x1 = "))
    except Exception:
        print("Error: numeros invalidos"); return
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}  x1={x1}")
    _secante_core(f, x0, x1, tol, maxit)


# ─── Newton-Raphson ───────────────────────────────────────────────────────────

def _newton_core(f, df, x0, tol, maxit):
    dfx_col = "f'(x_n)"
    print(f"\n{'Iter':<6} {'x_n':<22} {'f(x_n)':<22} {dfx_col:<22} {'EN':<15}")
    print("-" * 95)
    x = x0
    for it in range(maxit):
        try:
            fx, dfx = f(x), df(x)
        except Exception as e:
            print(f"\nError: {e}"); return None
        if not (math.isfinite(fx) and math.isfinite(dfx)):
            print("Error: f(x) o f'(x) no finito"); return None
        if abs(dfx) < 1e-15:
            print(f"Error: f'(x)~0 en x={x}"); return None
        xn = x - fx / dfx; EN = _en(xn, x)
        print(f"{it:<6d} {x:<22.10f} {fx:<22.10e} {dfx:<22.10e} {EN:<15.8e}")
        if EN < tol:
            fn = f(xn)
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}   f(x) = {f(x):.10e}")
    return x


def menu_newton_raphson():
    print("\n=== METODO DE NEWTON-RAPHSON ===\n")
    print("x_{n+1} = x_n - f(x_n)/f'(x_n)\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    try:
        df, df_str = derivada_simbolica(f_str); df(1)
        print(f"f'(x) = {df_str}\n")
    except Exception as e:
        print(f"Error calculando derivada: {e}"); return
    try:
        x0 = float(input("x0 = "))
    except Exception:
        print("Error: numero invalido"); return
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}")
    _newton_core(f, df, x0, tol, maxit)


# ─── Newton Mejorado (Halley) ─────────────────────────────────────────────────

def _newton_mejorado_core(f, df, d2f, x0, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n':<22} {'f(x_n)':<22} {'Paso':<22} {'EN':<15}")
    print("-" * 95)
    x = x0
    for it in range(maxit):
        try:
            fx, dfx, d2fx = f(x), df(x), d2f(x)
        except Exception as e:
            print(f"\nError: {e}"); return None
        if not (math.isfinite(fx) and math.isfinite(dfx) and math.isfinite(d2fx)):
            print("Error: valor no finito"); return None
        denom = dfx * dfx - fx * d2fx
        if abs(denom) < 1e-15:
            print(f"Error: denominador~0 en x={x}"); return None
        paso = (fx * dfx) / denom; xn = x - paso; EN = _en(xn, x)
        print(f"{it:<6d} {x:<22.10f} {fx:<22.10e} {paso:<22.10e} {EN:<15.8e}")
        if EN < tol:
            fn = f(xn)
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}   f(x) = {f(x):.10e}")
    return x


def menu_newton_mejorado():
    print("\n=== METODO DE NEWTON MEJORADO (HALLEY) ===\n")
    print("x_{n+1} = x_n - (f*f') / ((f')^2 - f*f'')\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    try:
        df, df_str, d2f, d2f_str = derivadas_simbolicas(f_str); df(1); d2f(1)
        print(f"f'(x)  = {df_str}")
        print(f"f''(x) = {d2f_str}\n")
    except Exception as e:
        print(f"Error calculando derivadas: {e}"); return
    try:
        x0 = float(input("x0 = "))
    except Exception:
        print("Error: numero invalido"); return
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}")
    _newton_mejorado_core(f, df, d2f, x0, tol, maxit)


# ─── Metodo Mixto ─────────────────────────────────────────────────────────────

_MM_MENU = {"1": "Biseccion", "2": "Regla Falsa", "3": "Newton-Raphson",
            "4": "Newton Mejorado", "5": "Secante", "6": "Punto Fijo"}
_MM_INTERVALO = {"1", "2"}
_MM_DERIVADA  = {"3", "4"}
_MM_G         = {"6"}


def _mm_actualizar(f, st, xn):
    try:
        fn = f(xn)
        if not math.isfinite(fn): return
        if st["fa"] * fn < 0: st["b"] = xn; st["fb"] = fn
        elif st["fb"] * fn < 0: st["a"] = xn; st["fa"] = fn
    except Exception:
        pass


def _mm_biseccion(f, st):
    a, b, fa, fb = st["a"], st["b"], st["fa"], st["fb"]
    if fa * fb > 0: raise ValueError("Sin cambio de signo en [a,b].")
    c = (a + b) / 2; fc = f(c)
    if fa * fc < 0: st["b"] = c; st["fb"] = fc
    else:            st["a"] = c; st["fa"] = fc
    return c, fc


def _mm_regla_falsa(f, st):
    a, b, fa, fb = st["a"], st["b"], st["fa"], st["fb"]
    if fa * fb > 0: raise ValueError("Sin cambio de signo en [a,b].")
    d = fb - fa
    if abs(d) < 1e-15: raise ValueError("f(b)-f(a)~0.")
    c = b - fb * (b - a) / d; fc = f(c)
    if fa * fc < 0: st["b"] = c; st["fb"] = fc
    else:            st["a"] = c; st["fa"] = fc
    return c, fc


def _mm_newton(f, df, st):
    x = st["x"]; fx = f(x); dfx = df(x)
    if not (math.isfinite(fx) and math.isfinite(dfx)): raise ValueError("No finito.")
    if abs(dfx) < 1e-15: raise ValueError("f'(x)~0.")
    xn = x - fx / dfx; fn = f(xn)
    _mm_actualizar(f, st, xn)
    return xn, fn


def _mm_newton2(f, df, d2f, st):
    x = st["x"]; fx = f(x); dfx = df(x); d2fx = d2f(x)
    if not all(math.isfinite(v) for v in [fx, dfx, d2fx]): raise ValueError("No finito.")
    d = dfx * dfx - fx * d2fx
    if abs(d) < 1e-15: raise ValueError("Denominador~0.")
    xn = x - (fx * dfx) / d; fn = f(xn)
    _mm_actualizar(f, st, xn)
    return xn, fn


def _mm_secante(f, st):
    x, xp = st["x"], st["x_prev"]
    fa, fb = f(x), f(xp)
    if not (math.isfinite(fa) and math.isfinite(fb)): raise ValueError("No finito.")
    d = fa - fb
    if abs(d) < 1e-15: raise ValueError("f(x_n)-f(x_{n-1})~0.")
    xn = x - fa * (x - xp) / d; fn = f(xn)
    _mm_actualizar(f, st, xn)
    return xn, fn


def _mm_punto_fijo(g, st):
    xn = g(st["x"])
    if not math.isfinite(xn): raise ValueError("g(x) diverge.")
    return xn, None


def _mixto_core(f, df, d2f, g, seq, st, tol, maxit):
    n = len(seq)
    print(f"\n{'Iter':<6} {'Metodo':<18} {'x_n':<22} {'x_n+1':<22} {'f(x_n+1)':<20} {'EN':<15}")
    print("-" * 108)
    for it in range(maxit):
        k = seq[it % n]; nom = _MM_MENU[k]; x_ant = st["x"]
        try:
            if k == "1": xn, fn = _mm_biseccion(f, st)
            elif k == "2": xn, fn = _mm_regla_falsa(f, st)
            elif k == "3": xn, fn = _mm_newton(f, df, st)
            elif k == "4": xn, fn = _mm_newton2(f, df, d2f, st)
            elif k == "5": xn, fn = _mm_secante(f, st)
            else:          xn, _  = _mm_punto_fijo(g, st); fn = f(xn)
        except Exception as e:
            print(f"\nError iter {it} ({nom}): {e}"); return None
        EN = _en(xn, x_ant)
        if it == 0:
            print(f"{it:<6} {nom:<18} {x_ant:<22.10f} {xn:<22.10f} {fn:<20.10e} {'---':<15}")
        else:
            print(f"{it:<6} {nom:<18} {x_ant:<22.10f} {xn:<22.10f} {fn:<20.10e} {EN:<15.8e}")
        st["x_prev"] = st["x"]; st["x"] = xn
        if it > 0 and EN < tol:
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {f(xn):.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn
    xf = st["x"]
    print(f"\nMax iter. Raiz aprox: x = {xf:.10f}   f(x) = {f(xf):.10e}")
    return xf


def menu_mixto():
    print("\n=== METODO MIXTO ===")
    print("Combina metodos alternandolos cada iteracion.\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nMetodos disponibles:")
    for k, v in _MM_MENU.items(): print(f"  {k}. {v}")
    raw = input("\nSecuencia (ej: '1 3'): ").strip().split()
    seq = [r for r in raw if r in _MM_MENU]
    if not seq: print("Secuencia invalida."); return
    print("Secuencia: " + " -> ".join(_MM_MENU[k] for k in seq) + " -> (ciclo)")
    claves = set(seq)
    df = d2f = g = None
    if claves & _MM_DERIVADA:
        try:
            df, df_str, d2f, d2f_str = derivadas_simbolicas(f_str); df(1); d2f(1)
            print(f"f'(x) = {df_str}   f''(x) = {d2f_str}")
        except Exception as e:
            print(f"Error derivadas: {e}"); return
    if claves & _MM_G:
        g_str = input("g(x) = ").strip()
        try:
            g = crear_funcion_segura(g_str); g(1)
        except Exception as e:
            print(f"Error g(x): {e}"); return
    print("\nIntervalo inicial [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    fa, fb = f(a), f(b)
    if (claves & _MM_INTERVALO) and fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(f, a, b); return
    x0_str = input(f"x0 (Enter={(a+b)/2:.6f}): ").strip()
    try:
        x0 = float(x0_str) if x0_str else (a + b) / 2
    except Exception:
        x0 = (a + b) / 2
    tol, maxit = _pedir_tol_iter()
    st = {"x": x0, "x_prev": a, "a": a, "b": b, "fa": fa, "fb": fb}
    print(f"\nResolviendo f(x)={f_str}  x0={x0}  [{a}, {b}]")
    _mixto_core(f, df, d2f, g, seq, st, tol, maxit)


def menu_raices():
    opciones = {
        "1": ("Bolzano",                  menu_bolzano),
        "2": ("Biseccion",                menu_biseccion),
        "3": ("Regla Falsa",              menu_regla_falsa),
        "4": ("Punto Fijo",               menu_punto_fijo),
        "5": ("Secante",                  menu_secante),
        "6": ("Newton-Raphson",           menu_newton_raphson),
        "7": ("Newton Mejorado (Halley)", menu_newton_mejorado),
        "8": ("Metodo Mixto",             menu_mixto),
    }
    print("\n=== RAICES DE ECUACIONES ===\n")
    for k, (v, _) in opciones.items(): print(f"  {k}. {v}")
    op = input("\nElige [1-8]: ").strip()
    if op in opciones:
        opciones[op][1]()
    else:
        print("Opcion invalida.")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMA 3: INTERPOLACION Y APROXIMACION
# ═══════════════════════════════════════════════════════════════════════════════

def _ingresar_datos_xy():
    print("\nFormato: valores separados por espacios  (ej: 1 2 3 4 5)\n")
    try:
        x = np.array([float(v) for v in input("x = ").strip().split()])
        y = np.array([float(v) for v in input("y = ").strip().split()])
    except Exception:
        print("Error: solo numeros separados por espacios."); return None, None
    if len(x) != len(y):
        print(f"Error: x tiene {len(x)} valores e y tiene {len(y)}."); return None, None
    if len(x) < 2:
        print("Error: minimo 2 puntos."); return None, None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        print("Error: NaN o Inf en los datos."); return None, None
    return x, y


def _print_tabla_xy(x, y):
    print(f"\n  {'i':>4}  {'xi':>14}  {'yi':>14}")
    print("  " + "-" * 36)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"  {i:>4}  {xi:>14.6g}  {yi:>14.6g}")


def _print_resultado(nombre, coefs_etiq, ec):
    print(f"\n{'='*58}")
    print(f"  {nombre}")
    print(f"{'='*58}")
    for etiq, val in coefs_etiq:
        print(f"  {etiq:>8} = {val:+.8g}")
    print(f"\n  Error cuadratico  Ec = {ec:.6e}")
    print(f"{'='*58}")


def regresion_lineal(x, y):
    n = len(x); sx = np.sum(x); sy = np.sum(y); sx2 = np.sum(x**2); sxy = np.sum(x*y)
    d = n * sx2 - sx**2
    if abs(d) < 1e-14: raise ValueError("Denominador nulo: datos constantes o colineales.")
    b1 = (n * sxy - sx * sy) / d
    b0 = (sx2 * sy - sxy * sx) / d
    ec = float(np.sum((y - (b0 + b1 * x))**2))
    return float(b0), float(b1), ec


def regresion_polinomial(x, y, grado):
    n = len(x)
    if grado < 1: raise ValueError("Grado debe ser >= 1.")
    if grado >= n: raise ValueError(f"Grado ({grado}) debe ser < numero de puntos ({n}).")
    A = np.vander(x, N=grado + 1, increasing=True)
    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ec = float(np.sum((y - A @ coefs)**2))
    return coefs, ec


def regresion_funcion_conocida(x, y, funciones_base):
    n = len(x); r = len(funciones_base)
    if r == 0: raise ValueError("Minimo 1 funcion base.")
    if r > n: raise ValueError(f"Funciones base ({r}) > datos ({n}).")
    Phi = np.column_stack([phi(x) for phi in funciones_base])
    if not np.all(np.isfinite(Phi)): raise ValueError("NaN/Inf en alguna funcion base.")
    coefs, _, rank, _ = np.linalg.lstsq(Phi, y, rcond=None)
    if rank < r: raise ValueError("Funciones base linealmente dependientes.")
    ec = float(np.sum((y - Phi @ coefs)**2))
    return coefs, ec


def regresion_exponencial(x, y):
    if np.any(y <= 0): raise ValueError("Todos los y deben ser > 0 para regresion exponencial.")
    Y = np.log(y); n = len(x); sx = np.sum(x); sY = np.sum(Y)
    sx2 = np.sum(x**2); sxY = np.sum(x * Y)
    d = n * sx2 - sx**2
    if abs(d) < 1e-14: raise ValueError("Denominador nulo.")
    B1 = (n * sxY - sx * sY) / d
    B0 = (sx2 * sY - sxY * sx) / d
    a = B1; b = math.exp(B0)
    ec = float(np.sum((y - b * np.exp(a * x))**2))
    return float(a), float(b), ec


def regresion_multiple(xv, yv, zv):
    """z = b0 + b1*x + b2*y  por minimos cuadrados."""
    n = len(xv)
    A = np.column_stack([np.ones(n), xv, yv])
    coefs, _, _, _ = np.linalg.lstsq(A, zv, rcond=None)
    ec = float(np.sum((zv - A @ coefs)**2))
    return coefs, ec


def _tabla_dif_div(x, y):
    n = len(x); T = np.zeros((n, n))
    T[:, 0] = y.copy()
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = (T[i+1, j-1] - T[i, j-1]) / (x[i+j] - x[i])
    return T


def _print_tabla_dif(x, y):
    T = _tabla_dif_div(x, y); n = len(x); w = 14
    enc = f"{'xi':>{w}} {'f(xi)':>{w}}"
    for k in range(1, n): enc += f" {'Orden '+str(k):>{w}}"
    print(enc); print("-" * w * (n + 1))
    for i in range(n):
        row = f"{x[i]:>{w}.6g} {T[i,0]:>{w}.6g}"
        for j in range(1, n - i): row += f" {T[i,j]:>{w}.6g}"
        print(row)
    print("\nCoeficientes de Newton (primera fila):")
    for k, c in enumerate(T[0, :]): print(f"  f[x0..x{k}] = {c:.8g}")


def interpolacion_newton(x, y, x_eval):
    T = _tabla_dif_div(x, y); coefs = T[0, :]; n = len(coefs)
    xe = np.asarray(x_eval, dtype=float)
    res = np.full_like(xe, coefs[n-1], dtype=float)
    for k in range(n-2, -1, -1):
        res = coefs[k] + (xe - x[k]) * res
    return res, coefs, T


def interpolacion_lagrange(x, y, x_eval):
    xe = np.atleast_1d(np.asarray(x_eval, dtype=float)); n = len(x)
    res = np.zeros_like(xe, dtype=float)
    p = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j: p[i] *= (x[i] - x[j])
    p = 1.0 / p
    for idx, xp in enumerate(xe):
        d = xp - x; ex = np.where(d == 0.0)[0]
        if len(ex) > 0: res[idx] = y[ex[0]]; continue
        t = p / d; res[idx] = np.dot(t, y) / np.sum(t)
    return res if res.size > 1 else float(res[0])


def _eval_poli(coefs, xp):
    return np.polyval(np.asarray(coefs)[::-1], np.asarray(xp, dtype=float))


def menu_interp_aprox():
    print("\n=== INTERPOLACION Y APROXIMACION (TEMA 3) ===\n")
    print("  1. Regresion lineal            y = b0 + b1*x")
    print("  2. Regresion polinomial        y = b0 + b1*x + ... + bm*x^m")
    print("  3. Regresion funcion conocida  F(x) = a0*phi0(x) + ...")
    print("  4. Regresion exponencial       y = b*exp(a*x)")
    print("  5. Regresion multiple 3D       z = b0 + b1*x + b2*y")
    print("  6. Interpolacion de Newton     (diferencias divididas)")
    print("  7. Interpolacion de Lagrange")
    try:
        op = int(input("\nElige [1-7]: ").strip())
    except Exception:
        print("Opcion invalida"); return
    if op not in range(1, 8):
        print("Opcion invalida"); return

    if op == 5:
        print("\nIngresa los puntos (x, y, z). Minimo 3 puntos.\n")
        try:
            xv = np.array([float(v) for v in input("x = ").strip().split()])
            yv = np.array([float(v) for v in input("y = ").strip().split()])
            zv = np.array([float(v) for v in input("z = ").strip().split()])
        except Exception:
            print("Error: solo numeros."); return
        if not (len(xv) == len(yv) == len(zv)):
            print("Error: x, y, z deben tener la misma longitud."); return
        if len(xv) < 3:
            print("Error: minimo 3 puntos."); return
        try:
            coefs, ec = regresion_multiple(xv, yv, zv)
        except Exception as e:
            print(f"Error: {e}"); return
        _print_resultado(
            "REGRESION MULTIPLE 3D\n  z = b0 + b1*x + b2*y",
            [("b0", coefs[0]), ("b1", coefs[1]), ("b2", coefs[2])], ec
        )
        ev = input("\nEvaluar en (x y) (Enter para omitir): ").strip()
        if ev:
            try:
                xp, yp = [float(v) for v in ev.split()]
                print(f"  F({xp}, {yp}) = {coefs[0]+coefs[1]*xp+coefs[2]*yp:.10g}")
            except Exception:
                print("Error: ingresa dos numeros separados por espacio.")
        return

    x, y = _ingresar_datos_xy()
    if x is None: return
    print("\nPuntos ingresados:")
    _print_tabla_xy(x, y)

    if op == 1:
        print("\n--- Regresion lineal ---")
        try:
            b0, b1, ec = regresion_lineal(x, y)
        except ValueError as e:
            print(f"Error: {e}"); return
        _print_resultado("REGRESION LINEAL   y = b0 + b1*x", [("b0", b0), ("b1", b1)], ec)
        ev = input("\nEvaluar en x (Enter para omitir): ").strip()
        if ev:
            try: xp = float(ev); print(f"  F({xp}) = {b0+b1*xp:.10g}")
            except Exception: print("Error: numero invalido.")

    elif op == 2:
        print("\n--- Regresion polinomial ---")
        try:
            m = int(input("Grado m = ").strip())
        except Exception:
            print("Error: entero invalido"); return
        try:
            coefs, ec = regresion_polinomial(x, y, m)
        except ValueError as e:
            print(f"Error: {e}"); return
        _print_resultado(f"REGRESION POLINOMIAL  grado {m}",
                         [(f"b{k}", coefs[k]) for k in range(len(coefs))], ec)
        ev = input("\nEvaluar en x (Enter para omitir): ").strip()
        if ev:
            try:
                xp = float(ev); print(f"  P({xp}) = {float(_eval_poli(coefs, xp)):.10g}")
            except Exception: print("Error: numero invalido.")

    elif op == 3:
        print("\n--- Regresion funcion conocida ---")
        print(AYUDA_FUNCIONES)
        print("\nEjemplos de funciones base: 1  |  x  |  x^2  |  sin(x)  |  exp(x)\n")
        try:
            r = int(input("Numero de funciones base r = ").strip())
            if r < 1: raise ValueError
        except Exception:
            print("Error: entero >= 1"); return
        funcs = []; nombres = []
        for k in range(r):
            phi_s = input(f"phi{k}(x) = ").strip()
            try:
                phi_e = crear_funcion_segura(phi_s); phi_e(1)
                phi_n = np.vectorize(phi_e); phi_n(np.array([1.0]))
            except Exception as e:
                print(f"Error phi{k}: {e}"); return
            funcs.append(phi_n); nombres.append(phi_s)
        try:
            coefs, ec = regresion_funcion_conocida(x, y, funcs)
        except ValueError as e:
            print(f"Error: {e}"); return
        modelo = " + ".join(f"a{k}*({nb})" for k, nb in enumerate(nombres))
        _print_resultado(f"REGRESION FUNCION CONOCIDA\n  F(x) = {modelo}",
                         [(f"a{k}", coefs[k]) for k in range(len(coefs))], ec)
        ev = input("\nEvaluar en x (Enter para omitir): ").strip()
        if ev:
            try:
                xp = float(ev); xa = np.array([xp])
                yp = sum(c * phi(xa) for c, phi in zip(coefs, funcs))
                print(f"  F({xp}) = {np.asarray(yp).ravel()[0]:.10g}")
            except Exception as e: print(f"Error: {e}")

    elif op == 4:
        print("\n--- Regresion exponencial ---")
        try:
            a, b_e, ec = regresion_exponencial(x, y)
        except ValueError as e:
            print(f"Error: {e}"); return
        _print_resultado("REGRESION EXPONENCIAL   y = b*exp(a*x)", [("a", a), ("b", b_e)], ec)
        ev = input("\nEvaluar en x (Enter para omitir): ").strip()
        if ev:
            try: xp = float(ev); print(f"  F({xp}) = {b_e*math.exp(a*xp):.10g}")
            except Exception: print("Error: numero invalido.")

    elif op == 6:
        if len(np.unique(x)) < len(x):
            print("Error: xi deben ser distintos para interpolacion."); return
        print("\n--- Interpolacion de Newton (diferencias divididas) ---")
        _print_tabla_dif(x, y)
        try:
            xp = float(input("\nEvaluar en x = ").strip())
        except Exception:
            print("Error: numero invalido."); return
        y_e, _, _ = interpolacion_newton(x, y, xp)
        print(f"\nP({xp}) = {float(y_e):.10g}")

    elif op == 7:
        if len(np.unique(x)) < len(x):
            print("Error: xi deben ser distintos para interpolacion."); return
        print("\n--- Interpolacion de Lagrange ---")
        try:
            xp = float(input("Evaluar en x = ").strip())
        except Exception:
            print("Error: numero invalido."); return
        y_e = interpolacion_lagrange(x, y, xp)
        print(f"\nP({xp}) = {float(y_e):.10g}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMA 4: DERIVACION E INTEGRACION NUMERICA
# ═══════════════════════════════════════════════════════════════════════════════

_TOL_NUM = 1e-12


def _leer_arr(nombre):
    t = input(f"{nombre} = ").strip()
    try:
        v = np.array([float(x) for x in t.split()], dtype=float)
    except ValueError as e:
        raise ValueError(f"Error en {nombre}: {e}")
    if v.size == 0: raise ValueError(f"Error: {nombre} vacio.")
    if not np.all(np.isfinite(v)): raise ValueError(f"Error: NaN/Inf en {nombre}.")
    return v


def _validar_ord(x, y):
    if x.size != y.size: raise ValueError("x e y deben tener igual longitud.")
    if x.size < 3: raise ValueError("Minimo 3 puntos.")
    o = np.argsort(x); xo, yo = x[o], y[o]
    if np.any(np.isclose(np.diff(xo), 0, atol=_TOL_NUM, rtol=0)):
        raise ValueError("xi deben ser distintos.")
    return xo, yo


def _calc_h(x):
    h = np.diff(x); h0 = float(h[0])
    es_cte = np.allclose(h, h0, atol=1e-10, rtol=1e-8)
    return h, h0, es_cte, float(np.max(np.abs(h - h0)))


def _polinomio_lag(x, y):
    n = x.size; p = np.poly1d([0.0])
    for i in range(n):
        b = np.poly1d([1.0]); d = 1.0
        for j in range(n):
            if i == j: continue
            b *= np.poly1d([1.0, -x[j]]); d *= (x[i] - x[j])
        p += y[i] * (b / d)
    return p


def _fmt_poly(p, var="x"):
    c = np.asarray(p.c, dtype=float); g = len(c) - 1; terms = []
    for k, cf in enumerate(c):
        ex = g - k
        if abs(cf) < _TOL_NUM: continue
        sg = "-" if cf < 0 else "+"; m = abs(cf); ms = f"{m:.10g}"
        if ex == 0: t = ms
        elif ex == 1: t = var if np.isclose(m, 1, atol=_TOL_NUM, rtol=0) else f"{ms}*{var}"
        else: t = (f"{var}^{ex}" if np.isclose(m, 1, atol=_TOL_NUM, rtol=0) else f"{ms}*{var}^{ex}")
        terms.append((sg, t))
    if not terms: return "0"
    ps = []
    for i, (sg, t) in enumerate(terms):
        ps.append(t if (i == 0 and sg == "+") else (f"-{t}" if i == 0 else f" {sg} {t}"))
    return "".join(ps)


# ─── 1a derivada ──────────────────────────────────────────────────────────────

def _d1_hcte(x, y, esq, h):
    n = x.size; res = []
    if esq == "adelante":
        for i in range(n - 2):
            d = (-3*y[i] + 4*y[i+1] - y[i+2]) / (2*h)
            res.append({"i": i, "x": x[i], "d": d, "h1": h, "h2": h, "nodos": f"x{i},x{i+1},x{i+2}"})
    elif esq == "central":
        for i in range(1, n - 1):
            d = (y[i+1] - y[i-1]) / (2*h)
            res.append({"i": i, "x": x[i], "d": d, "h1": h, "h2": h, "nodos": f"x{i-1},x{i},x{i+1}"})
    else:
        for i in range(2, n):
            d = (3*y[i] - 4*y[i-1] + y[i-2]) / (2*h)
            res.append({"i": i, "x": x[i], "d": d, "h1": h, "h2": h, "nodos": f"x{i-2},x{i-1},x{i}"})
    return res


def _d1_hvar(x, y, esq):
    n = x.size; res = []
    if esq == "adelante":
        for i in range(n - 2):
            h1 = x[i+1]-x[i]; h2 = x[i+2]-x[i+1]
            c0 = -(2*h1+h2)/(h1*(h1+h2)); c1 = (h1+h2)/(h1*h2); c2 = -h1/(h2*(h1+h2))
            d = c0*y[i] + c1*y[i+1] + c2*y[i+2]
            res.append({"i": i, "x": x[i], "d": d, "h1": h1, "h2": h2, "nodos": f"x{i},x{i+1},x{i+2}"})
    elif esq == "central":
        for i in range(1, n - 1):
            h1 = x[i]-x[i-1]; h2 = x[i+1]-x[i]
            ci = -h2/(h1*(h1+h2)); c0 = (h2-h1)/(h1*h2); cd = h1/(h2*(h1+h2))
            d = ci*y[i-1] + c0*y[i] + cd*y[i+1]
            res.append({"i": i, "x": x[i], "d": d, "h1": h1, "h2": h2, "nodos": f"x{i-1},x{i},x{i+1}"})
    else:
        for i in range(2, n):
            h1 = x[i]-x[i-1]; h2 = x[i-1]-x[i-2]
            c0 = (2*h1+h2)/(h1*(h1+h2)); c1 = -(h1+h2)/(h1*h2); c2 = h1/(h2*(h1+h2))
            d = c0*y[i] + c1*y[i-1] + c2*y[i-2]
            res.append({"i": i, "x": x[i], "d": d, "h1": h1, "h2": h2, "nodos": f"x{i-2},x{i-1},x{i}"})
    return res


def _print_d1(res, dp):
    print("\nResultados — 1a derivada:")
    print(f"  {'i':>4} {'x_i':>16} {'f_aprox':>18} {'f_Lag':>18} {'|Error|':>14} {'h1':>12} {'h2':>12} {'Nodos':>16}")
    print("  " + "-" * 118)
    for r in res:
        xi = r["x"]; da = float(r["d"]); dl = float(dp(xi)); err = abs(da - dl)
        if abs(da) < _TOL_NUM: da = 0.0
        if abs(dl) < _TOL_NUM: dl = 0.0
        if err < _TOL_NUM: err = 0.0
        print(f"  {r['i']:>4} {xi:>16.10g} {da:>18.10g} {dl:>18.10g} {err:>14.6e} {r['h1']:>12.10g} {r['h2']:>12.10g} {r['nodos']:>16}")


# ─── 2a derivada ──────────────────────────────────────────────────────────────

def _d2_hcte(x, y, h):
    """f''(xi) = (f(xi-1) - 2f(xi) + f(xi+1)) / h^2"""
    n = x.size; res = []
    for i in range(1, n - 1):
        d = (y[i-1] - 2*y[i] + y[i+1]) / h**2
        res.append({"i": i, "x": x[i], "d": d})
    return res


def _d2_hvar(x, y):
    """2a derivada del polinomio Lagrange local de 3 puntos: 2a derivada en xi."""
    n = x.size; res = []
    for i in range(1, n - 1):
        h1 = x[i] - x[i-1]; h2 = x[i+1] - x[i]
        d = 2 * (y[i-1]/(h1*(h1+h2)) - y[i]/(h1*h2) + y[i+1]/(h2*(h1+h2)))
        res.append({"i": i, "x": x[i], "d": d})
    return res


def _print_d2(res, d2p):
    print("\nResultados — 2a derivada:")
    print(f"  {'i':>4} {'x_i':>16} {'f2_aprox':>18} {'f2_Lag':>18} {'|Error|':>14}")
    print("  " + "-" * 78)
    for r in res:
        xi = r["x"]; da = float(r["d"]); dl = float(d2p(xi)); err = abs(da - dl)
        print(f"  {r['i']:>4} {xi:>16.10g} {da:>18.10g} {dl:>18.10g} {err:>14.6e}")


# ─── 3a derivada (h constante, 5 puntos) ─────────────────────────────────────

def _d3_hcte(x, y, h):
    """f'''(xi) = (-f(xi-2) + 2f(xi-1) - 2f(xi+1) + f(xi+2)) / (2h^3)"""
    n = x.size; res = []
    for i in range(2, n - 2):
        d = (-y[i-2] + 2*y[i-1] - 2*y[i+1] + y[i+2]) / (2 * h**3)
        res.append({"i": i, "x": x[i], "d": d})
    return res


def _print_d3(res, d3p):
    print("\nResultados — 3a derivada:")
    print(f"  {'i':>4} {'x_i':>16} {'f3_aprox':>18} {'f3_Lag':>18} {'|Error|':>14}")
    print("  " + "-" * 78)
    for r in res:
        xi = r["x"]; da = float(r["d"]); dl = float(d3p(xi)); err = abs(da - dl)
        print(f"  {r['i']:>4} {xi:>16.10g} {da:>18.10g} {dl:>18.10g} {err:>14.6e}")


# ─── 4a derivada (h constante, 5 puntos) ─────────────────────────────────────

def _d4_hcte(x, y, h):
    """f''''(xi) = (f(xi-2) - 4f(xi-1) + 6f(xi) - 4f(xi+1) + f(xi+2)) / h^4"""
    n = x.size; res = []
    for i in range(2, n - 2):
        d = (y[i-2] - 4*y[i-1] + 6*y[i] - 4*y[i+1] + y[i+2]) / h**4
        res.append({"i": i, "x": x[i], "d": d})
    return res


def _print_d4(res, d4p):
    print("\nResultados — 4a derivada:")
    print(f"  {'i':>4} {'x_i':>16} {'f4_aprox':>18} {'f4_Lag':>18} {'|Error|':>14}")
    print("  " + "-" * 78)
    for r in res:
        xi = r["x"]; da = float(r["d"]); dl = float(d4p(xi)); err = abs(da - dl)
        print(f"  {r['i']:>4} {xi:>16.10g} {da:>18.10g} {dl:>18.10g} {err:>14.6e}")


# ─── Modulo derivacion desde puntos ──────────────────────────────────────────

def _print_tabla_h(x, h):
    print("\nTabla de pasos h_i (pares consecutivos):")
    print(f"  {'i':>4} {'x_i':>16} {'x_(i+1)':>16} {'h_i':>16} {'h_i - h_0':>16}")
    print("  " + "-" * 76)
    h0 = h[0]
    for i, hi in enumerate(h):
        print(f"  {i:>4} {x[i]:>16.10g} {x[i+1]:>16.10g} {hi:>16.10g} {hi-h0:>16.10g}")


def _resolver_modo_h(es_cte, modo, h):
    h_prom = float(np.mean(h))
    aviso = None
    if modo == 1:
        usar_cte = es_cte
    elif modo == 2:
        usar_cte = True
        if not es_cte:
            aviso = f"Se forzo h constante pero los datos no son uniformes. Se usara h promedio = {h_prom:.10g}."
    else:
        usar_cte = False
        if es_cte:
            aviso = "Se forzo h no constante aunque los datos son uniformes. Se aplicaran formulas generales."
    return usar_cte, h_prom, aviso


def _pedir_modo_h():
    print("\nModo de trabajo para h:")
    print("  1. Detectar automaticamente si h es constante")
    print("  2. Forzar h constante")
    print("  3. Forzar h no constante")
    try: modo = int(input("Elige [1-3]: ").strip())
    except Exception: modo = 1
    return modo if modo in (1, 2, 3) else 1


def _pedir_esquema():
    print("\nEsquema de derivacion:")
    print("  1. Hacia adelante")
    print("  2. Central")
    print("  3. Hacia atras")
    try: e = int(input("Elige [1-3]: ").strip())
    except Exception: e = 2
    return {1: "adelante", 2: "central", 3: "atras"}.get(e, "central")


def _modulo_derivacion_puntos():
    print("\n=== DERIVACION DESDE PUNTOS ===\n")
    print("  1. Primera derivada")
    print("  2. Segunda derivada")
    print("  3. Tercera derivada  (h constante, minimo 5 puntos)")
    print("  4. Cuarta derivada   (h constante, minimo 5 puntos)")
    try:
        sub = int(input("\nElige [1-4]: ").strip())
    except Exception:
        print("Opcion invalida"); return
    if sub not in (1, 2, 3, 4):
        print("Opcion invalida"); return

    try:
        x = _leer_arr("x"); y = _leer_arr("y")
        x, y = _validar_ord(x, y)
    except ValueError as e:
        print(e); return

    # Tabla de puntos
    print(f"\nPuntos ordenados ({x.size} puntos):")
    print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
    print("  " + "-" * 40)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")

    # Tabla de pasos h
    h, h0, es_cte, err_max = _calc_h(x)
    _print_tabla_h(x, h)

    # Deteccion automatica
    print("\nDeteccion automatica de h:")
    if es_cte:
        print(f"  h constante detectado  (h = {h0:.10g})")
    else:
        print(f"  h no constante detectado.")
        print(f"  h_0 = {h0:.10g},  desviacion maxima = {err_max:.6e}")

    # Polinomio de Lagrange global (referencia)
    p_lag = _polinomio_lag(x, y)
    print(f"\nPolinomio interpolante de Lagrange global:")
    print(f"  P_{x.size-1}(x) = {_fmt_poly(p_lag)}")

    if sub == 1:
        modo = _pedir_modo_h()
        usar_cte, h_prom, aviso = _resolver_modo_h(es_cte, modo, h)
        esq = _pedir_esquema()

        tipo_h = "constante" if usar_cte else "no constante"
        origen = {1: "automatico", 2: "forzado", 3: "forzado"}[modo]
        print(f"\nConfiguracion elegida:")
        print(f"  Tipo de h : {tipo_h} ({origen})")
        print(f"  Esquema   : {esq}")
        if aviso: print(f"  Aviso     : {aviso}")

        dp = np.polyder(p_lag)
        print(f"\nDerivada del polinomio global:")
        print(f"  P_{x.size-1}'(x) = {_fmt_poly(dp)}")

        h_usar = h0 if (usar_cte and es_cte) else h_prom
        res = _d1_hcte(x, y, esq, h_usar) if usar_cte else _d1_hvar(x, y, esq)
        if not res:
            print("No hay suficientes nodos para el esquema elegido."); return
        _print_d1(res, dp)

    elif sub == 2:
        modo = _pedir_modo_h()
        usar_cte, h_prom, aviso = _resolver_modo_h(es_cte, modo, h)

        tipo_h = "constante" if usar_cte else "no constante"
        origen = {1: "automatico", 2: "forzado", 3: "forzado"}[modo]
        print(f"\nConfiguracion elegida:")
        print(f"  Tipo de h : {tipo_h} ({origen})")
        if aviso: print(f"  Aviso     : {aviso}")

        d2p = np.polyder(p_lag, 2)
        print(f"\nSegunda derivada del polinomio global:")
        print(f"  P_{x.size-1}''(x) = {_fmt_poly(d2p)}")

        h_usar = h0 if (usar_cte and es_cte) else h_prom
        res = _d2_hcte(x, y, h_usar) if usar_cte else _d2_hvar(x, y)
        if not res:
            print("No hay nodos interiores."); return
        _print_d2(res, d2p)

    elif sub == 3:
        if x.size < 5:
            print("Error: minimo 5 puntos para 3a derivada."); return
        h_usar = h0
        if not es_cte:
            h_usar = float(np.mean(h))
            print(f"Aviso: h no constante. Se usara h promedio = {h_usar:.10g}.")
        d3p = np.polyder(p_lag, 3)
        print(f"\nTercera derivada del polinomio global:")
        print(f"  P_{x.size-1}'''(x) = {_fmt_poly(d3p)}")
        res = _d3_hcte(x, y, h_usar)
        if not res:
            print("No hay nodos validos."); return
        _print_d3(res, d3p)

    elif sub == 4:
        if x.size < 5:
            print("Error: minimo 5 puntos para 4a derivada."); return
        h_usar = h0
        if not es_cte:
            h_usar = float(np.mean(h))
            print(f"Aviso: h no constante. Se usara h promedio = {h_usar:.10g}.")
        d4p = np.polyder(p_lag, 4)
        print(f"\nCuarta derivada del polinomio global:")
        print(f"  P_{x.size-1}''''(x) = {_fmt_poly(d4p)}")
        res = _d4_hcte(x, y, h_usar)
        if not res:
            print("No hay nodos validos."); return
        _print_d4(res, d4p)


# ─── Modulo derivacion desde polinomio ────────────────────────────────────────

def _modulo_polinomio():
    print("\n=== DERIVACION DESDE POLINOMIO ===\n")
    print("Ingresa coeficientes en orden descendente  (ej: 1 -3 2  =>  x^2 - 3x + 2)")
    try:
        c = np.array([float(v) for v in input("coeficientes = ").strip().split()])
    except Exception:
        print("Error: numeros invalidos."); return
    if c.size == 0:
        print("Error: ingresa al menos un coeficiente."); return
    p = np.poly1d(c); dp = np.polyder(p)
    print(f"\nP(x)  = {_fmt_poly(p)}")
    print(f"P'(x) = {_fmt_poly(dp)}")
    try:
        xv = _leer_arr("Puntos x para evaluar")
    except ValueError as e:
        print(e); return
    yv  = np.where(np.abs(p(xv)) < _TOL_NUM, 0.0, p(xv))
    dyv = np.where(np.abs(dp(xv)) < _TOL_NUM, 0.0, dp(xv))
    dp_col = "P'(x_i)"
    print(f"\n  {'i':>4} {'x_i':>16} {'P(x_i)':>16} {dp_col:>16}")
    print("  " + "-" * 60)
    for i, (xi, yi, dyi) in enumerate(zip(xv, yv, dyv)):
        print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g} {dyi:>16.10g}")
    xu, idx = np.unique(xv, return_index=True); yu = yv[idx]
    if xu.size >= 2:
        p_rec = _polinomio_lag(xu, yu); dp_rec = np.polyder(p_rec)
        print(f"\nPolinomio recuperado (Lagrange):  P_rec(x) = {_fmt_poly(p_rec)}")
        print(f"Derivada:  P_rec'(x) = {_fmt_poly(dp_rec)}")


# ─── Integracion numerica ─────────────────────────────────────────────────────

def _trapecio(x, y):
    """Trapecio compuesto (funciona con h variable)."""
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


def _simpson13(x, y):
    """Simpson 1/3 compuesto. Requiere n par y h constante."""
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par (numero de subintervalos).")
    h, h0, es_cte, _ = _calc_h(x)
    if not es_cte:
        raise ValueError("Simpson 1/3 requiere h constante (datos uniformes).")
    pesos = np.ones(n + 1)
    pesos[1:-1:2] = 4
    pesos[2:-2:2] = 2
    return float(h0 / 3 * np.dot(pesos, y))


def _simpson38(x, y):
    """Simpson 3/8 compuesto. Requiere n multiplo de 3 y h constante."""
    n = len(x) - 1
    if n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n multiplo de 3.")
    h, h0, es_cte, _ = _calc_h(x)
    if not es_cte:
        raise ValueError("Simpson 3/8 requiere h constante (datos uniformes).")
    pesos = np.ones(n + 1)
    for i in range(1, n):
        pesos[i] = 2 if i % 3 == 0 else 3
    return float(3 * h0 / 8 * np.dot(pesos, y))


def _modulo_integracion():
    print("\n=== INTEGRACION NUMERICA ===\n")
    print("  1. Trapecio compuesto          (h variable permitido)")
    print("  2. Simpson 1/3 compuesto       (n par, h constante)")
    print("  3. Simpson 3/8 compuesto       (n multiplo de 3, h constante)")
    try:
        sub = int(input("\nElige [1-3]: ").strip())
    except Exception:
        print("Opcion invalida"); return
    if sub not in (1, 2, 3):
        print("Opcion invalida"); return

    print("\nIngresa los puntos tabulares.\n")
    try:
        x = _leer_arr("x"); y = _leer_arr("y")
    except ValueError as e:
        print(e); return
    if x.size != y.size:
        print("Error: x e y deben tener igual longitud."); return
    if x.size < 2:
        print("Error: minimo 2 puntos."); return
    orden = np.argsort(x); x = x[orden]; y = y[orden]
    if np.any(np.isclose(np.diff(x), 0, atol=_TOL_NUM, rtol=0)):
        print("Error: xi deben ser distintos."); return

    n_sub = len(x) - 1
    print(f"\n{n_sub} subintervalos, {len(x)} puntos.  a={x[0]:.6g}  b={x[-1]:.6g}")

    if sub == 1:
        r = _trapecio(x, y)
        print(f"\nRegla del Trapecio:  integral ≈ {r:.10g}")
    elif sub == 2:
        try:
            r = _simpson13(x, y)
            print(f"\nSimpson 1/3:  integral ≈ {r:.10g}")
        except ValueError as e:
            print(f"Error: {e}")
    elif sub == 3:
        try:
            r = _simpson38(x, y)
            print(f"\nSimpson 3/8:  integral ≈ {r:.10g}")
        except ValueError as e:
            print(f"Error: {e}")


def menu_derivacion_integracion():
    print("\n=== DERIVACION E INTEGRACION NUMERICA (TEMA 4) ===\n")
    print("  1. Derivacion desde puntos  (1a, 2a, 3a, 4a derivada)")
    print("  2. Derivacion desde polinomio ingresado")
    print("  3. Integracion numerica     (Trapecio, Simpson 1/3, Simpson 3/8)")
    try:
        op = int(input("\nElige [1-3]: ").strip())
    except Exception:
        print("Opcion invalida"); return
    if op == 1:   _modulo_derivacion_puntos()
    elif op == 2: _modulo_polinomio()
    elif op == 3: _modulo_integracion()
    else:         print("Opcion invalida")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMA 5: EDOs
# ═══════════════════════════════════════════════════════════════════════════════

_AYUDA_XY = (
    "Funciones disponibles (variables: x, y):\n"
    "  sin, cos, tan, exp, log (=ln), sqrt, abs\n"
    "  asin, acos, atan, sinh, cosh, tanh, log2, log10\n"
    "Constantes: pi, e\n"
    "Potencias: usa ^ o **\n"
    "Ejemplos:  y - x  |  x*y  |  -2*y  |  sin(x) + cos(y)"
)


def _pedir_ode():
    print(_AYUDA_XY)
    print("\nIngresa f(x, y) tal que  y' = f(x, y)\n")
    f_str = input("f(x,y) = ").strip()
    try:
        f = crear_funcion_xy_segura(f_str); f(0, 1)
    except Exception as e:
        print(f"Error: {e}"); return None, None
    return f, f_str


def _pedir_ci():
    try:
        x0 = float(input("x0 = "))
        y0 = float(input("y0 = "))
        xf = float(input("x final = "))
        if xf <= x0:
            print("Error: x final debe ser > x0"); return None
    except Exception:
        print("Error: numeros invalidos."); return None
    try:
        h_str = input("Paso h (Enter para especificar n): ").strip()
        if h_str:
            h = float(h_str)
            if h <= 0: print("Error: h debe ser > 0"); return None
        else:
            n = int(input("Numero de pasos n = ").strip())
            if n <= 0: print("Error: n debe ser > 0"); return None
            h = (xf - x0) / n
    except Exception:
        print("Error: valor invalido."); return None
    return x0, y0, xf, h


def _edo_header(cols):
    print("\n" + f"{'n':<6} " + " ".join(f"{c:<20}" for c in cols))
    print("-" * (6 + 21 * len(cols)))


def metodo_euler(f, x0, y0, xf, h):
    print("\n--- Euler:  y_{n+1} = y_n + h*f(x_n, y_n) ---")
    _edo_header(["x_n", "y_n", "f(x_n,y_n)", "y_{n+1}"])
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try: k = f(x, y)
        except Exception as e: print(f"Error: {e}"); return None
        yn = y + h_ef * k
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return y


def metodo_rk2_pm(f, x0, y0, xf, h):
    print("\n--- RK2 Punto Medio:  k1=f(x,y), k2=f(x+h/2, y+h/2*k1), y_{n+1}=y+h*k2 ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "y_{n+1}"])
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x, y); k2 = f(x + h_ef/2, y + h_ef/2 * k1)
        except Exception as e: print(f"Error: {e}"); return None
        yn = y + h_ef * k2
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return y


def metodo_rk2_heun(f, x0, y0, xf, h):
    print("\n--- RK2 Heun:  k1=f(x,y), k2=f(x+h, y+h*k1), y_{n+1}=y+(h/2)*(k1+k2) ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "y_{n+1}"])
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x, y); k2 = f(x + h_ef, y + h_ef * k1)
        except Exception as e: print(f"Error: {e}"); return None
        yn = y + (h_ef / 2) * (k1 + k2)
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return y


def metodo_rk4(f, x0, y0, xf, h):
    print("\n--- RK4:  y_{n+1} = y_n + (h/6)*(k1 + 2k2 + 2k3 + k4) ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "k3", "k4", "y_{n+1}"])
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x,            y)
            k2 = f(x + h_ef/2,   y + h_ef/2 * k1)
            k3 = f(x + h_ef/2,   y + h_ef/2 * k2)
            k4 = f(x + h_ef,     y + h_ef   * k3)
        except Exception as e: print(f"Error: {e}"); return None
        yn = y + (h_ef / 6) * (k1 + 2*k2 + 2*k3 + k4)
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {k3:<20.10g} {k4:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return y


def menu_odes():
    print("\n=== ECUACIONES DIFERENCIALES ORDINARIAS (TEMA 5) ===\n")
    print("  1. Metodo de Euler")
    print("  2. RK2 Punto Medio")
    print("  3. RK2 Heun  (Trapecio)")
    print("  4. RK4  (Runge-Kutta orden 4)")
    try:
        op = int(input("\nElige [1-4]: ").strip())
    except Exception:
        print("Opcion invalida"); return
    if op not in (1, 2, 3, 4):
        print("Opcion invalida"); return
    f, f_str = _pedir_ode()
    if f is None: return
    ci = _pedir_ci()
    if ci is None: return
    x0, y0, xf, h = ci
    print(f"\ny' = {f_str},  y({x0}) = {y0},  x ∈ [{x0}, {xf}],  h = {h}")
    if op == 1:   metodo_euler(f, x0, y0, xf, h)
    elif op == 2: metodo_rk2_pm(f, x0, y0, xf, h)
    elif op == 3: metodo_rk2_heun(f, x0, y0, xf, h)
    elif op == 4: metodo_rk4(f, x0, y0, xf, h)


# ═══════════════════════════════════════════════════════════════════════════════
# MENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    while True:
        print("\n" + "═" * 60)
        print("  METODOS NUMERICOS — MENU PRINCIPAL")
        print("═" * 60)
        print("  1. Raices de ecuaciones")
        print("  2. Interpolacion y Aproximacion   (Tema 3)")
        print("  3. Derivacion e Integracion       (Tema 4)")
        print("  4. Ecuaciones Diferenciales       (Tema 5)")
        print("  0. Salir")
        print("═" * 60)
        try:
            op = input("Elige [0-4]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego."); break
        if op == "0":
            print("Hasta luego."); break
        elif op == "1": menu_raices()
        elif op == "2": menu_interp_aprox()
        elif op == "3": menu_derivacion_integracion()
        elif op == "4": menu_odes()
        else:
            print("Opcion invalida. Elige entre 0 y 4.")


if __name__ == "__main__":
    main()
