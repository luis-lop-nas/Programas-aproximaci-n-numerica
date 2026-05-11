"""tema_raices.py — Raices de ecuaciones."""

from common import *

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
    n = _pedir_int("Subdivisiones (Enter=1000): ", condicion=lambda v: v > 0,
                   error="subdivisiones debe ser > 0", permitir_vacio=True, default=1000)
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
    if abs(fa) < tol:
        print(f"\nRaiz exacta en a: x = {a:.10f}   f(x) = {fa:.10e}")
        return a, [a], [0.0]
    if abs(fb) < tol:
        print(f"\nRaiz exacta en b: x = {b:.10f}   f(x) = {fb:.10e}")
        return b, [b], [0.0]
    if fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(funcion, a, b); return None, [], []
    print(f"\n{'Iter':<6} {'a':<18} {'b':<18} {'c':<18} {'f(c)':<18} {'EN':<15}")
    print("-" * 95)
    hist_x = []; hist_en = []
    c_ant = None
    for it in range(maxit):
        c = (a + b) / 2; fc = funcion(c)
        EN = _en(c, c_ant) if c_ant is not None else float('inf')
        hist_x.append(c)
        hist_en.append(EN if math.isfinite(EN) else None)
        if c_ant is None:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {EN:<15.8e}")
        if abs(fc) < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {0.0:.10e}   Iter: {it+1}")
            hist_en[-1] = 0.0
            return c, hist_x, hist_en
        if c_ant is not None and EN < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return c, hist_x, hist_en
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        c_ant = c
    c = (a + b) / 2; fc = funcion(c)
    hist_x.append(c)
    print(f"\nMax iter. Raiz aprox: x = {c:.10f}   f(x) = {fc:.10e}")
    return c, hist_x, hist_en


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
    raiz, hist_x, hist_en = _biseccion_core(f, a, b, tol, maxit)
    if raiz is None: return
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str, 'a': a, 'b': b,
                        'raiz': raiz, 'metodo': 'Biseccion', 'hist_en': hist_en})


# ─── Regla Falsa ──────────────────────────────────────────────────────────────

def _regla_falsa_core(funcion, a, b, tol, maxit):
    fa, fb = funcion(a), funcion(b)
    if abs(fa) < tol:
        print(f"\nRaiz exacta en a: x = {a:.10f}   f(x) = {fa:.10e}")
        return a, [a], [0.0]
    if abs(fb) < tol:
        print(f"\nRaiz exacta en b: x = {b:.10f}   f(x) = {fb:.10e}")
        return b, [b], [0.0]
    if fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(funcion, a, b); return None, [], []
    print(f"\n{'Iter':<6} {'a':<18} {'b':<18} {'c':<18} {'f(c)':<18} {'EN':<15}")
    print("-" * 95)
    hist_x = []; hist_en = []
    c_ant = c = None
    for it in range(maxit):
        d = fb - fa
        if abs(d) < 1e-15:
            print("Error: f(b)-f(a)~0"); return None, [], []
        c = b - fb * (b - a) / d; fc = funcion(c)
        EN = _en(c, c_ant) if c_ant is not None else float('inf')
        hist_x.append(c)
        hist_en.append(EN if math.isfinite(EN) else None)
        if c_ant is None:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {EN:<15.8e}")
        if abs(fc) < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {0.0:.10e}   Iter: {it+1}")
            hist_en[-1] = 0.0
            return c, hist_x, hist_en
        if c_ant is not None and EN < tol:
            print(f"\nRaiz: x = {c:.10f}   f(x) = {fc:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return c, hist_x, hist_en
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        c_ant = c
    print(f"\nMax iter. Raiz aprox: x = {c:.10f}   f(x) = {funcion(c):.10e}")
    return c, hist_x, hist_en


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
    raiz, hist_x, hist_en = _regla_falsa_core(f, a, b, tol, maxit)
    if raiz is None: return
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str, 'a': a, 'b': b,
                        'raiz': raiz, 'metodo': 'Regla Falsa', 'hist_en': hist_en})


# ─── Punto Fijo ───────────────────────────────────────────────────────────────

def _punto_fijo_core(f, g, x0, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n':<22} {'x_n+1':<22} {'f(x_n+1)':<22} {'EN':<15}")
    print("-" * 92)
    hist_x = [x0]; hist_en = []
    x = x0
    for it in range(maxit):
        try:
            xn = g(x)
        except Exception as e:
            print(f"\nError g(x): {e}"); return None, [], []
        if not math.isfinite(xn):
            print("Error: g(x) diverge"); return None, [], []
        try:
            fn = f(xn)
        except Exception:
            fn = float('nan')
        EN = _en(xn, x)
        hist_x.append(xn); hist_en.append(EN)
        if it == 0:
            print(f"{it:<6} {x:<22.10f} {xn:<22.10f} {fn:<22.10e} {'---':<15}")
        else:
            print(f"{it:<6} {x:<22.10f} {xn:<22.10f} {fn:<22.10e} {EN:<15.8e}")
        if it > 0 and EN < tol:
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn, hist_x, hist_en
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}")
    try:
        print(f"f({x:.10f}) = {f(x):.10e}")
    except Exception:
        pass
    return x, hist_x, hist_en


def menu_punto_fijo():
    print("\n=== METODO DE PUNTO FIJO ===\n")
    print("x_{n+1} = g(x_n)  donde  f(x)=0  se reescribe como  x=g(x)\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    while True:
        g_str = _pedir_texto("g(x) = ")
        try:
            g = crear_funcion_segura(g_str); g(1)
            break
        except Exception as e:
            print(f"Error g(x): {e}")
    x0 = _pedir_float("\nx0 = ")
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  g(x)={g_str}  x0={x0}")
    raiz, hist_x, hist_en = _punto_fijo_core(f, g, x0, tol, maxit)
    if raiz is None: return
    a_plot = min(hist_x) - abs(min(hist_x)) * 0.1 - 0.5
    b_plot = max(hist_x) + abs(max(hist_x)) * 0.1 + 0.5
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str,
                        'a': a_plot, 'b': b_plot,
                        'raiz': raiz, 'metodo': 'Punto Fijo', 'hist_en': hist_en})


# ─── Secante ──────────────────────────────────────────────────────────────────

def _secante_core(f, x0, x1, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n-1':<22} {'x_n':<22} {'x_n+1':<22} {'f(x_n+1)':<18} {'EN':<15}")
    print("-" * 110)
    hist_x = [x0, x1]; hist_en = []
    xa, xb = x0, x1
    for it in range(maxit):
        try:
            fa, fb = f(xa), f(xb)
        except Exception as e:
            print(f"\nError: {e}"); return None, [], []
        if not (math.isfinite(fa) and math.isfinite(fb)):
            print("Error: f(x) no finito"); return None, [], []
        d = fb - fa
        if abs(d) < 1e-15:
            print(f"Error: f(x_n)-f(x_{{n-1}})~0 en iter {it}"); return None, [], []
        xn = xb - fb * (xb - xa) / d
        if not math.isfinite(xn):
            print("Error: x diverge"); return None, [], []
        fn = f(xn); EN = _en(xn, xb)
        hist_x.append(xn); hist_en.append(EN)
        if it == 0:
            print(f"{it:<6} {xa:<22.10f} {xb:<22.10f} {xn:<22.10f} {fn:<18.10e} {'---':<15}")
        else:
            print(f"{it:<6} {xa:<22.10f} {xb:<22.10f} {xn:<22.10f} {fn:<18.10e} {EN:<15.8e}")
        if it > 0 and EN < tol:
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn, hist_x, hist_en
        xa, xb = xb, xn
    print(f"\nMax iter. Raiz aprox: x = {xb:.10f}   f(x) = {f(xb):.10e}")
    return xb, hist_x, hist_en


def menu_secante():
    print("\n=== METODO DE LA SECANTE ===\n")
    print("x_{n+1} = x_n - f(x_n)*(x_n-x_{n-1}) / (f(x_n)-f(x_{n-1}))\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    x0 = _pedir_float("x0 = ")
    x1 = _pedir_float("x1 = ")
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}  x1={x1}")
    raiz, hist_x, hist_en = _secante_core(f, x0, x1, tol, maxit)
    if raiz is None: return
    xs_sorted = sorted(hist_x)
    pad = max(abs(xs_sorted[0]), abs(xs_sorted[-1])) * 0.2 + 0.5
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str,
                        'a': xs_sorted[0] - pad, 'b': xs_sorted[-1] + pad,
                        'raiz': raiz, 'metodo': 'Secante', 'hist_en': hist_en})


# ─── Newton-Raphson ───────────────────────────────────────────────────────────

def _newton_core(f, df, x0, tol, maxit):
    dfx_col = "f'(x_n)"
    print(f"\n{'Iter':<6} {'x_n':<22} {'f(x_n)':<22} {dfx_col:<22} {'EN':<15}")
    print("-" * 95)
    hist_x = [x0]; hist_en = []
    x = x0
    for it in range(maxit):
        try:
            fx, dfx = f(x), df(x)
        except Exception as e:
            print(f"\nError: {e}"); return None, [], []
        if not (math.isfinite(fx) and math.isfinite(dfx)):
            print("Error: f(x) o f'(x) no finito"); return None, [], []
        if abs(dfx) < 1e-15:
            print(f"Error: f'(x)~0 en x={x}"); return None, [], []
        xn = x - fx / dfx; EN = _en(xn, x)
        hist_x.append(xn); hist_en.append(EN)
        print(f"{it:<6d} {x:<22.10f} {fx:<22.10e} {dfx:<22.10e} {EN:<15.8e}")
        if EN < tol:
            fn = f(xn)
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn, hist_x, hist_en
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}   f(x) = {f(x):.10e}")
    return x, hist_x, hist_en


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
    x0 = _pedir_float("x0 = ")
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}")
    raiz, hist_x, hist_en = _newton_core(f, df, x0, tol, maxit)
    if raiz is None: return
    xs_sorted = sorted(hist_x)
    pad = max(abs(xs_sorted[0]), abs(xs_sorted[-1])) * 0.2 + 0.5
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str,
                        'a': xs_sorted[0] - pad, 'b': xs_sorted[-1] + pad,
                        'raiz': raiz, 'metodo': 'Newton-Raphson', 'hist_en': hist_en})


# ─── Newton Mejorado (Halley) ─────────────────────────────────────────────────

def _newton_mejorado_core(f, df, d2f, x0, tol, maxit):
    print(f"\n{'Iter':<6} {'x_n':<22} {'f(x_n)':<22} {'Paso':<22} {'EN':<15}")
    print("-" * 95)
    hist_x = [x0]; hist_en = []
    x = x0
    for it in range(maxit):
        try:
            fx, dfx, d2fx = f(x), df(x), d2f(x)
        except Exception as e:
            print(f"\nError: {e}"); return None, [], []
        if not (math.isfinite(fx) and math.isfinite(dfx) and math.isfinite(d2fx)):
            print("Error: valor no finito"); return None, [], []
        denom = dfx * dfx - fx * d2fx
        if abs(denom) < 1e-15:
            print(f"Error: denominador~0 en x={x}"); return None, [], []
        paso = (fx * dfx) / denom; xn = x - paso; EN = _en(xn, x)
        hist_x.append(xn); hist_en.append(EN)
        print(f"{it:<6d} {x:<22.10f} {fx:<22.10e} {paso:<22.10e} {EN:<15.8e}")
        if EN < tol:
            fn = f(xn)
            print(f"\nRaiz: x = {xn:.10f}   f(x) = {fn:.10e}   EN = {EN:.10e}   Iter: {it+1}")
            return xn, hist_x, hist_en
        x = xn
    print(f"\nMax iter. Raiz aprox: x = {x:.10f}   f(x) = {f(x):.10e}")
    return x, hist_x, hist_en


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
    x0 = _pedir_float("x0 = ")
    tol, maxit = _pedir_tol_iter()
    print(f"\nResolviendo f(x)={f_str}  x0={x0}")
    raiz, hist_x, hist_en = _newton_mejorado_core(f, df, d2f, x0, tol, maxit)
    if raiz is None: return
    xs_sorted = sorted(hist_x)
    pad = max(abs(xs_sorted[0]), abs(xs_sorted[-1])) * 0.2 + 0.5
    _preguntar_grafica({'tipo': 'raiz', 'f': f, 'f_str': f_str,
                        'a': xs_sorted[0] - pad, 'b': xs_sorted[-1] + pad,
                        'raiz': raiz, 'metodo': 'Newton Mejorado', 'hist_en': hist_en})


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
    raw = _input("\nSecuencia (ej: '1 3'): ").strip().split()
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
        while True:
            g_str = _pedir_texto("g(x) = ")
            try:
                g = crear_funcion_segura(g_str); g(1)
                break
            except Exception as e:
                print(f"Error g(x): {e}")
    print("\nIntervalo inicial [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    fa, fb = f(a), f(b)
    if (claves & _MM_INTERVALO) and fa * fb > 0:
        print(f"\nNo hay cambio de signo en [{a}, {b}].")
        sugerir_intervalos(f, a, b); return
    x0 = _pedir_float(f"x0 (Enter={(a+b)/2:.6f}): ", permitir_vacio=True, default=(a + b) / 2)
    tol, maxit = _pedir_tol_iter()
    st = {"x": x0, "x_prev": a, "a": a, "b": b, "fa": fa, "fb": fb}
    print(f"\nResolviendo f(x)={f_str}  x0={x0}  [{a}, {b}]")
    _mixto_core(f, df, d2f, g, seq, st, tol, maxit)


# ─── Polinomios con coeficientes enteros (Tema 10) ───────────────────────────

def _divisores(n):
    n = abs(n)
    if n == 0:
        return []
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _division_sintetica(coefs, c):
    b = [coefs[0]]
    for i in range(1, len(coefs)):
        b.append(coefs[i] + c * b[-1])
    return b[:-1], b[-1]


def _contar_cambios_signo(coefs):
    no_cero = [v for v in coefs if v != 0]
    return sum(1 for i in range(len(no_cero) - 1) if no_cero[i] * no_cero[i + 1] < 0)


def _poly_str(coefs):
    n = len(coefs) - 1
    partes = []
    for i, c in enumerate(coefs):
        exp = n - i
        if c == 0:
            continue
        if exp == 0:
            partes.append(f"{c:+g}")
        elif exp == 1:
            partes.append(f"{c:+g}x")
        else:
            partes.append(f"{c:+g}x^{exp}")
    if not partes:
        return "0"
    s = " ".join(partes)
    return s[1:] if s.startswith("+") else s


def _horner_poly(coefs, x):
    r = coefs[0]
    for c in coefs[1:]:
        r = r * x + c
    return r


def menu_polinomios_enteros():
    print("\n=== RAICES DE POLINOMIOS CON COEFICIENTES ENTEROS ===\n")
    print("Coeficientes en orden DESCENDENTE (a_n ... a_1 a_0)")
    print("Ej. para 2x^3 + 4x^2 - 22x - 24:  2 4 -22 -24\n")
    try:
        coefs_orig = [int(v) for v in _input("Coeficientes: ").strip().split()]
    except Exception:
        print("Error: introduce enteros separados por espacios."); return
    if len(coefs_orig) < 2:
        print("Error: minimo 2 coeficientes."); return
    if coefs_orig[0] == 0:
        print("Error: el coeficiente lider no puede ser cero."); return

    grado_orig = len(coefs_orig) - 1
    print(f"\nP(x) = {_poly_str(coefs_orig)}  (grado {grado_orig})")

    # ── Raices nulas ──────────────────────────────────────────────────────────
    coefs = list(coefs_orig)
    raices_nulas = 0
    while len(coefs) > 1 and coefs[-1] == 0:
        raices_nulas += 1
        coefs = coefs[:-1]
    if raices_nulas:
        print(f"\n  Raices nulas: x = 0  (multiplicidad {raices_nulas})")
        print(f"  Polinomio reducido: {_poly_str(coefs)}")
    if len(coefs) == 1:
        print("\nPolinomio es trivial tras factorizar raices nulas."); return

    # ── Regla de Descartes ────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("REGLA DE LOS SIGNOS DE DESCARTES")
    print(f"{'─'*62}")
    n_deg = len(coefs) - 1
    cambios_pos = _contar_cambios_signo(coefs)
    coefs_neg = [coefs[i] * ((-1) ** (n_deg - i)) for i in range(len(coefs))]
    cambios_neg = _contar_cambios_signo(coefs_neg)

    posibles_pos = [cambios_pos - 2 * k for k in range(cambios_pos // 2 + 1)]
    posibles_neg = [cambios_neg - 2 * k for k in range(cambios_neg // 2 + 1)]

    print(f"\n  P(x)  = {_poly_str(coefs)}")
    print(f"  Cambios de signo: {cambios_pos}  →  raices positivas: {', '.join(map(str, posibles_pos))}")
    print(f"\n  P(-x) = {_poly_str(coefs_neg)}")
    print(f"  Cambios de signo: {cambios_neg}  →  raices negativas: {', '.join(map(str, posibles_neg))}")

    # ── Teorema de la raiz racional ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("TEOREMA DE LA RAIZ RACIONAL")
    print(f"{'─'*62}")
    divs_b = _divisores(coefs[-1])
    divs_c = _divisores(coefs[0])
    print(f"\n  a_0 = {coefs[-1]}  →  |divisores|: {divs_b}")
    print(f"  a_n = {coefs[0]}  →  |divisores|: {divs_c}")

    candidatos = set()
    for b in divs_b:
        for c in divs_c:
            g = math.gcd(b, c)
            p, q = b // g, c // g
            candidatos.add((p, q))
            candidatos.add((-p, q))
    candidatos = sorted(candidatos, key=lambda x: (x[1] != 1, abs(x[0] / x[1]), x[0]))

    def _frac(p, q):
        return str(p) if q == 1 else f"{p}/{q}"

    print(f"\n  Candidatos: {', '.join(_frac(p, q) for p, q in candidatos)}")

    # ── Tabulacion ────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("TABULACION")
    print(f"{'─'*62}")
    print(f"\n  {'Candidato':>10}  {'P(r)':>16}  Raiz?")
    print(f"  {'─'*36}")
    raices_exactas = []
    for p, q in candidatos:
        r = p / q
        val = _horner_poly(coefs, r)
        es_raiz = "SI" if abs(val) < 1e-8 else ""
        if es_raiz:
            raices_exactas.append((p, q, r))
        val_s = f"{val:.6g}" if abs(val) < 1e15 else "overflow"
        print(f"  {_frac(p, q):>10}  {val_s:>16}  {es_raiz}")

    # ── Division sintetica ────────────────────────────────────────────────────
    if not raices_exactas:
        print("\nNo se encontraron raices racionales entre los candidatos.")
        if raices_nulas:
            print(f"\nRaiz: x = 0  (multiplicidad {raices_nulas})")
        return

    print(f"\n{'─'*62}")
    print("DIVISION SINTETICA")
    print(f"{'─'*62}")

    coefs_actual = list(coefs)
    raices_encontradas = []

    for p, q, r in raices_exactas:
        if len(coefs_actual) <= 1:
            break
        if abs(_horner_poly(coefs_actual, r)) > 1e-8:
            continue

        print(f"\n  P(x) = {_poly_str(coefs_actual)}  ÷  (x - {_frac(p, q)})\n")

        n_c = len(coefs_actual)
        col = 10
        print("        " + "".join(f"{c:>{col}g}" for c in coefs_actual))

        cociente, resto = _division_sintetica(coefs_actual, r)
        prods = [None] + [r * cociente[i] for i in range(n_c - 1)]

        prod_str = f"  {_frac(p, q):>4} |" + "".join(
            ("" if v is None else f"{v:>{col}g}") for v in prods
        )
        print(prod_str)
        print("        " + "─" * (col * n_c))

        print("        " + "".join(f"{v:>{col}g}" for v in cociente)
              + f"  | r = {resto:g}")
        print(f"\n  Cociente: {_poly_str(cociente)}   Resto: {resto:g}")

        raices_encontradas.append(r)
        coefs_actual = cociente

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("RESUMEN")
    print(f"{'═'*62}")
    if raices_nulas:
        print(f"  x = 0  (multiplicidad {raices_nulas})")
    for r in raices_encontradas:
        print(f"  x = {r:g}")
    if len(coefs_actual) > 1:
        grado_res = len(coefs_actual) - 1
        print(f"\n  Residual (grado {grado_res}): {_poly_str(coefs_actual)}")
        if grado_res == 1:
            r_lin = -coefs_actual[1] / coefs_actual[0]
            print(f"  → raiz adicional: x = {r_lin:g}")
            raices_encontradas.append(r_lin)
        elif grado_res == 2:
            a2, b2, c2 = coefs_actual
            disc = b2**2 - 4 * a2 * c2
            if disc >= 0:
                r1 = (-b2 + disc**0.5) / (2 * a2)
                r2 = (-b2 - disc**0.5) / (2 * a2)
                print(f"  → discriminante = {disc:g}  →  x = {r1:g},  x = {r2:g}")
            else:
                print(f"  → discriminante = {disc:g}  →  raices complejas")
        else:
            print("  (puede tener raices irracionales o complejas)")
    total = raices_nulas + len(raices_encontradas)
    print(f"\n  Raices racionales halladas: {total} / {grado_orig}")
    print(f"{'═'*62}")


def menu_graficar_raices():
    if not _MPLOK:
        print("  matplotlib no disponible."); return
    print("\n=== GRAFICAR — RAICES ===\n")
    print("  1. f(x) con raiz marcada (un metodo)")
    print("  2. Comparar metodos (f(x) + convergencia)")
    print("  0. Volver")
    try:
        op = _input("\nElige [0-2]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    if op == "0": return

    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo para graficar [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return

    _METODOS_RAIZ = {
        "1": ("Biseccion",        lambda f, a, b, tol, mit: _biseccion_core(f, a, b, tol, mit)),
        "2": ("Regla Falsa",      lambda f, a, b, tol, mit: _regla_falsa_core(f, a, b, tol, mit)),
        "3": ("Newton-Raphson",   None),
        "4": ("Secante",          None),
    }

    if op == "1":
        print("\nMetodo:")
        for k, (nm, _) in _METODOS_RAIZ.items(): print(f"  {k}. {nm}")
        try:
            sel = _input("Elige [1-4]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            return
        tol, maxit = _pedir_tol_iter()
        raiz = hist_en = None
        if sel in ("1", "2"):
            fn = _METODOS_RAIZ[sel][1]
            r, _, hen = fn(f, a, b, tol, maxit)
            raiz = r; hist_en = hen
        elif sel == "3":
            try:
                df, _ = derivada_simbolica(f_str); df(1)
            except Exception as e:
                print(f"Error derivada: {e}"); return
            x0 = _pedir_float("x0 = ")
            r, _, hen = _newton_core(f, df, x0, tol, maxit)
            raiz = r; hist_en = hen
        elif sel == "4":
            try:
                x0 = _pedir_float("x0 = "); x1 = _pedir_float("x1 = ")
            except VolverAtras: return
            r, _, hen = _secante_core(f, x0, x1, tol, maxit)
            raiz = r; hist_en = hen
        if raiz is None: return
        _gr.raiz_fx(f, a, b, raiz, f_str, _METODOS_RAIZ.get(sel, ("?",))[0])

    elif op == "2":
        print("\nMetodos a comparar (varios con espacio, ej: 1 2):")
        print("  1. Biseccion   2. Regla Falsa   3. Newton-Raphson   4. Secante")
        try:
            raw = _input("Metodos: ").strip().split()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            return
        sels = [r for r in raw if r in ("1", "2", "3", "4")]
        if not sels: sels = ["1", "2"]
        tol, maxit = _pedir_tol_iter()
        datos_list = []
        df = None
        if "3" in sels:
            try:
                df, _ = derivada_simbolica(f_str); df(1)
            except Exception as e:
                print(f"Error derivada: {e}"); sels = [s for s in sels if s != "3"]
        x0_sec = x1_sec = None
        if "4" in sels:
            try:
                x0_sec = _pedir_float("x0 para Secante = ")
                x1_sec = _pedir_float("x1 para Secante = ")
            except VolverAtras:
                sels = [s for s in sels if s != "4"]
        nombres = {"1": "Biseccion", "2": "Regla Falsa", "3": "Newton-Raphson", "4": "Secante"}
        for s in sels:
            if s == "1":
                r, _, hen = _biseccion_core(f, a, b, tol, maxit)
            elif s == "2":
                r, _, hen = _regla_falsa_core(f, a, b, tol, maxit)
            elif s == "3":
                try: x0_nw = _pedir_float(f"x0 para Newton: ")
                except VolverAtras: continue
                r, _, hen = _newton_core(f, df, x0_nw, tol, maxit)
            elif s == "4":
                r, _, hen = _secante_core(f, x0_sec, x1_sec, tol, maxit)
            else:
                continue
            datos_list.append({'metodo': nombres[s], 'raiz': r, 'hist_en': hen or []})
        if datos_list:
            _gr.raiz_comparacion(f, a, b, f_str, datos_list)


def menu_raices():
    opciones = {
        "1": ("Bolzano",                             menu_bolzano),
        "2": ("Biseccion",                           menu_biseccion),
        "3": ("Regla Falsa",                         menu_regla_falsa),
        "4": ("Punto Fijo",                          menu_punto_fijo),
        "5": ("Secante",                             menu_secante),
        "6": ("Newton-Raphson",                      menu_newton_raphson),
        "7": ("Newton Mejorado (Halley)",            menu_newton_mejorado),
        "8": ("Metodo Mixto",                        menu_mixto),
        "9": ("Polinomios coef. enteros (Tema 10)",  menu_polinomios_enteros),
    }
    while True:
        print("\n=== RAICES DE ECUACIONES ===\n")
        for k, (v, _) in opciones.items(): print(f"  {k}. {v}")
        print(" 10. Graficar")
        print("  0. Volver al menu principal")
        try:
            op = _input("\nElige [0-10]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op == "0":
            break
        elif op == "10":
            try: menu_graficar_raices()
            except VolverAtras: continue
        elif op in opciones:
            try: opciones[op][1]()
            except VolverAtras: continue
        else:
            print("Opcion invalida.")




__all__ = [name for name in globals() if not name.startswith("__")]
