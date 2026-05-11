"""tema_derivacion_integracion.py — Derivacion e integracion numerica."""

from common import *

# ═══════════════════════════════════════════════════════════════════════════════
# TEMA 4: DERIVACION E INTEGRACION NUMERICA
# ═══════════════════════════════════════════════════════════════════════════════

_TOL_NUM = 1e-12


def _leer_arr(nombre):
    while True:
        t = _input(f"{nombre} = ").strip()
        try:
            v = np.array([float(x) for x in t.split()], dtype=float)
        except ValueError as e:
            print(f"Error en {nombre}: {e}")
            continue
        if v.size == 0:
            print(f"Error: {nombre} vacio.")
            continue
        if not np.all(np.isfinite(v)):
            print(f"Error: NaN/Inf en {nombre}.")
            continue
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


# ─── Aproximacion en un punto desde funcion ──────────────────────────────────

_ESQUEMAS_D1_PT = {
    "adelante": [
        ("2 puntos  O(h)",
         lambda f, x0, h: (f(x0 + h) - f(x0)) / h,
         "[f(x0+h) - f(x0)] / h"),
        ("3 puntos  O(h^2)",
         lambda f, x0, h: (-3*f(x0) + 4*f(x0 + h) - f(x0 + 2*h)) / (2*h),
         "[-3 f(x0) + 4 f(x0+h) - f(x0+2h)] / (2h)"),
    ],
    "central": [
        ("2 puntos  O(h^2)",
         lambda f, x0, h: (f(x0 + h) - f(x0 - h)) / (2*h),
         "[f(x0+h) - f(x0-h)] / (2h)"),
        ("4 puntos  O(h^4)",
         lambda f, x0, h: (f(x0 - 2*h) - 8*f(x0 - h) + 8*f(x0 + h) - f(x0 + 2*h)) / (12*h),
         "[f(x0-2h) - 8 f(x0-h) + 8 f(x0+h) - f(x0+2h)] / (12 h)"),
    ],
    "atras": [
        ("2 puntos  O(h)",
         lambda f, x0, h: (f(x0) - f(x0 - h)) / h,
         "[f(x0) - f(x0-h)] / h"),
        ("3 puntos  O(h^2)",
         lambda f, x0, h: (3*f(x0) - 4*f(x0 - h) + f(x0 - 2*h)) / (2*h),
         "[3 f(x0) - 4 f(x0-h) + f(x0-2h)] / (2h)"),
    ],
}

_ESQUEMAS_D2_PT = {
    "adelante": [
        ("3 puntos  O(h)",
         lambda f, x0, h: (f(x0) - 2*f(x0 + h) + f(x0 + 2*h)) / h**2,
         "[f(x0) - 2 f(x0+h) + f(x0+2h)] / h^2"),
        ("4 puntos  O(h^2)",
         lambda f, x0, h: (2*f(x0) - 5*f(x0 + h) + 4*f(x0 + 2*h) - f(x0 + 3*h)) / h**2,
         "[2 f(x0) - 5 f(x0+h) + 4 f(x0+2h) - f(x0+3h)] / h^2"),
    ],
    "central": [
        ("3 puntos  O(h^2)",
         lambda f, x0, h: (f(x0 - h) - 2*f(x0) + f(x0 + h)) / h**2,
         "[f(x0-h) - 2 f(x0) + f(x0+h)] / h^2"),
        ("5 puntos  O(h^4)",
         lambda f, x0, h: (-f(x0 - 2*h) + 16*f(x0 - h) - 30*f(x0) + 16*f(x0 + h) - f(x0 + 2*h)) / (12*h**2),
         "[-f(x0-2h) + 16 f(x0-h) - 30 f(x0) + 16 f(x0+h) - f(x0+2h)] / (12 h^2)"),
    ],
    "atras": [
        ("3 puntos  O(h)",
         lambda f, x0, h: (f(x0) - 2*f(x0 - h) + f(x0 - 2*h)) / h**2,
         "[f(x0) - 2 f(x0-h) + f(x0-2h)] / h^2"),
        ("4 puntos  O(h^2)",
         lambda f, x0, h: (2*f(x0) - 5*f(x0 - h) + 4*f(x0 - 2*h) - f(x0 - 3*h)) / h**2,
         "[2 f(x0) - 5 f(x0-h) + 4 f(x0-2h) - f(x0-3h)] / h^2"),
    ],
}


def _aproximar_derivada_punto(orden):
    """orden: 1 -> primera derivada, 2 -> segunda derivada."""
    titulo = "PRIMERA" if orden == 1 else "SEGUNDA"
    etiqueta = "f'" if orden == 1 else "f''"
    print(f"\n=== APROXIMACION DE LA {titulo} DERIVADA EN UN PUNTO ===\n")
    print(AYUDA_FUNCIONES)

    while True:
        f_str = _pedir_texto("\nf(x) = ")
        try:
            f = crear_funcion_segura(f_str)
            f(1.0)
            break
        except Exception as e:
            print(f"Error en la funcion: {e}")

    x0 = _pedir_float("x0 = ")
    h = _pedir_float("h  = ", condicion=lambda v: v > 0, error="h debe ser > 0")

    esq = _pedir_esquema()
    tabla = _ESQUEMAS_D1_PT if orden == 1 else _ESQUEMAS_D2_PT
    formulas = tabla[esq]

    print("\nOrden de precision:")
    for i, (nombre, _, _) in enumerate(formulas, 1):
        print(f"  {i}. {nombre}")
    op = _pedir_int(f"Elige [1-{len(formulas)}] (Enter=1): ", condicion=lambda v: 1 <= v <= len(formulas),
                    error=f"elige entre 1 y {len(formulas)}", permitir_vacio=True, default=1)
    nombre, calc, fmla = formulas[op - 1]

    try:
        valor = float(calc(f, x0, h))
    except Exception as e:
        print(f"Error evaluando f en los nodos: {e}"); return

    if orden == 1:
        offsets = {"adelante": [0, 1, 2], "central": [-2, -1, 1, 2], "atras": [-2, -1, 0]}[esq]
    else:
        offsets = {"adelante": [0, 1, 2, 3], "central": [-2, -1, 0, 1, 2], "atras": [-3, -2, -1, 0]}[esq]

    deriv_str = None; deriv_val = None
    try:
        if orden == 1:
            df, deriv_str = derivada_simbolica(f_str)
            deriv_val = float(df(x0))
        else:
            _, _, d2f, deriv_str = derivadas_simbolicas(f_str)
            deriv_val = float(d2f(x0))
    except Exception:
        pass

    print(f"\nConfiguracion:")
    print(f"  Esquema  : {esq}")
    print(f"  Precision: {nombre}")
    print(f"  Formula  : {etiqueta}(x0) ≈ {fmla}")
    print(f"  x0 = {x0:.10g},  h = {h:.10g}")

    print(f"\nNodos usados:")
    print(f"  {'k':>4} {'x_k':>16} {'f(x_k)':>18}")
    print("  " + "-" * 42)
    for k in offsets:
        xk = x0 + k*h
        try: fk = float(f(xk))
        except Exception: fk = float("nan")
        print(f"  {k:>4} {xk:>16.10g} {fk:>18.10g}")

    print(f"\nResultado:")
    print(f"  {etiqueta}({x0:.10g}) ≈ {valor:.10g}")
    if deriv_val is not None:
        err = abs(valor - deriv_val)
        print(f"  {etiqueta}_exacta(x) = {deriv_str}")
        print(f"  {etiqueta}_exacta({x0:.10g}) = {deriv_val:.10g}")
        print(f"  |Error| = {err:.6e}")


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
    print("  2. Forzar h constante  (usa h promedio con los y originales)")
    print("  3. Forzar h no constante")
    print("  4. h constante via interpolacion local  (remuestrea en malla uniforme)")
    return _pedir_int("Elige [1-4] (Enter=1): ", condicion=lambda v: 1 <= v <= 4,
                      error="elige entre 1 y 4", permitir_vacio=True, default=1)


def _interpolar_uniforme(x_orig, y_orig):
    """
    Remuestrea (x_orig, y_orig) en una malla uniforme usando interpolacion local
    de Lagrange con los 3 puntos originales mas cercanos a cada nodo nuevo.
    Pide h al usuario (por defecto h = h_prom).
    Devuelve (x_uni, y_uni, h_uni).
    """
    h_prom = float(np.mean(np.diff(x_orig)))
    print(f"\n  Interpolacion local a malla uniforme:")
    print(f"  Sugerencia h = h_prom = {h_prom:.10g}")
    h_uni = _pedir_float("  h (Enter para usar h_prom): ", condicion=lambda v: v > 0,
                         error="h debe ser > 0", permitir_vacio=True, default=h_prom)

    x_a, x_b = float(x_orig[0]), float(x_orig[-1])
    n_int = max(2, round((x_b - x_a) / h_uni))
    h_uni = (x_b - x_a) / n_int          # ajuste exacto para cubrir [x_a, x_b]
    x_uni = np.array([x_a + i * h_uni for i in range(n_int + 1)])

    n_orig = x_orig.size
    y_uni = np.zeros(x_uni.size)
    for k, xk in enumerate(x_uni):
        dists = np.abs(x_orig - xk)
        idx3 = np.sort(np.argsort(dists)[:min(3, n_orig)])
        x3 = x_orig[idx3]; y3 = y_orig[idx3]
        val = 0.0
        for i in range(len(x3)):
            L = 1.0
            for j in range(len(x3)):
                if i != j:
                    L *= (xk - x3[j]) / (x3[i] - x3[j])
            val += float(y3[i]) * L
        y_uni[k] = val

    return x_uni, y_uni, h_uni


def _pedir_esquema():
    print("\nEsquema de derivacion:")
    print("  1. Hacia adelante")
    print("  2. Central")
    print("  3. Hacia atras")
    e = _pedir_int("Elige [1-3] (Enter=2): ", condicion=lambda v: 1 <= v <= 3,
                   error="elige entre 1 y 3", permitir_vacio=True, default=2)
    return {1: "adelante", 2: "central", 3: "atras"}.get(e, "central")


def _aproximar_derivada_punto_tabla(orden):
    """Aproxima la n-esima derivada en un nodo concreto eligiendo esquema adelante/central/atras."""
    titulo = "PRIMERA" if orden == 1 else "SEGUNDA"
    etiqueta = "f'" if orden == 1 else "f''"
    print(f"\n=== APROXIMACION DE LA {titulo} DERIVADA EN UN PUNTO (DESDE TABLA) ===\n")

    try:
        x = _leer_arr("x"); y = _leer_arr("y")
        x, y = _validar_ord(x, y)
    except ValueError as e:
        print(e); return

    n = x.size
    print(f"\nPuntos ordenados ({n} puntos):")
    print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
    print("  " + "-" * 40)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")

    h_arr, h0, es_cte, _ = _calc_h(x)
    _print_tabla_h(x, h_arr)

    idx = _pedir_int(f"\nIndice i del nodo a aproximar [0-{n-1}]: ",
                     condicion=lambda v: 0 <= v < n,
                     error=f"indice fuera de rango [0, {n-1}]")

    esq = _pedir_esquema()

    # Validacion segun esquema (ambas derivadas necesitan 3 nodos)
    if esq == "adelante" and idx + 2 >= n:
        print(f"Error: adelante en i={idx} necesita hasta i={idx+2} (solo hay {n} puntos)."); return
    if esq == "central" and (idx == 0 or idx == n - 1):
        print(f"Error: central necesita vecinos a ambos lados (i=0 e i={n-1} no validos)."); return
    if esq == "atras" and idx - 2 < 0:
        print(f"Error: atras en i={idx} necesita desde i={idx-2}."); return

    modo = _pedir_modo_h()

    # Referencia Lagrange global sobre datos originales
    p_lag = _polinomio_lag(x, y)
    dp_ref = np.polyder(p_lag, orden)
    der_sym = "'" * orden
    print(f"\n  P_{n-1}{der_sym}(x) = {_fmt_poly(dp_ref)}")

    # ── Modo 4: remuestrear en malla uniforme ──────────────────────────────────
    if modo == 4:
        x_c, y_c, h_usar = _interpolar_uniforme(x, y)
        n_c = x_c.size
        print(f"\n  Malla uniforme ({n_c} puntos, h = {h_usar:.10g}):")
        print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
        print("  " + "-" * 40)
        for i, (xi, yi) in enumerate(zip(x_c, y_c)):
            print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")

        # nodo de la malla uniforme mas cercano al x[idx] solicitado
        idx_c = int(np.argmin(np.abs(x_c - x[idx])))
        print(f"\n  Nodo mas cercano en malla uniforme: i={idx_c},  x={x_c[idx_c]:.10g}  (solicitado: x={x[idx]:.10g})")

        if esq == "adelante" and idx_c + 2 >= n_c:
            print(f"Error: adelante en i={idx_c} necesita hasta i={idx_c+2} (malla tiene {n_c} puntos)."); return
        if esq == "central" and (idx_c == 0 or idx_c == n_c - 1):
            print(f"Error: central necesita vecinos a ambos lados en la malla uniforme."); return
        if esq == "atras" and idx_c - 2 < 0:
            print(f"Error: atras en i={idx_c} necesita desde i={idx_c-2}."); return

        hh = h_usar
        if orden == 1:
            if esq == "adelante":
                d = (-3*y_c[idx_c] + 4*y_c[idx_c+1] - y_c[idx_c+2]) / (2*hh)
                nodos_str = f"x{idx_c}, x{idx_c+1}, x{idx_c+2}  (malla uniforme)"
                formula_str = "[-3 f + 4 f(+h) - f(+2h)] / (2h)"
            elif esq == "central":
                d = (y_c[idx_c+1] - y_c[idx_c-1]) / (2*hh)
                nodos_str = f"x{idx_c-1}, x{idx_c}, x{idx_c+1}  (malla uniforme)"
                formula_str = "[f(+h) - f(-h)] / (2h)"
            else:
                d = (3*y_c[idx_c] - 4*y_c[idx_c-1] + y_c[idx_c-2]) / (2*hh)
                nodos_str = f"x{idx_c-2}, x{idx_c-1}, x{idx_c}  (malla uniforme)"
                formula_str = "[3 f - 4 f(-h) + f(-2h)] / (2h)"
        else:
            if esq == "adelante":
                d = (y_c[idx_c] - 2*y_c[idx_c+1] + y_c[idx_c+2]) / hh**2
                nodos_str = f"x{idx_c}, x{idx_c+1}, x{idx_c+2}  (malla uniforme)"
                formula_str = "[f - 2 f(+h) + f(+2h)] / h^2"
            elif esq == "central":
                d = (y_c[idx_c-1] - 2*y_c[idx_c] + y_c[idx_c+1]) / hh**2
                nodos_str = f"x{idx_c-1}, x{idx_c}, x{idx_c+1}  (malla uniforme)"
                formula_str = "[f(-h) - 2 f + f(+h)] / h^2"
            else:
                d = (y_c[idx_c] - 2*y_c[idx_c-1] + y_c[idx_c-2]) / hh**2
                nodos_str = f"x{idx_c-2}, x{idx_c-1}, x{idx_c}  (malla uniforme)"
                formula_str = "[f - 2 f(-h) + f(-2h)] / h^2"

        x_eval = x_c[idx_c]
        print(f"\nConfiguracion:")
        print(f"  Tipo de h : constante (interpolacion local Lagrange)")
        print(f"  h = {h_usar:.10g},  Esquema : {esq}")

    # ── Modos 1-3: datos originales ────────────────────────────────────────────
    else:
        usar_cte, h_prom, aviso = _resolver_modo_h(es_cte, modo, h_arr)
        h_usar = h0 if (usar_cte and es_cte) else h_prom
        tipo_h = "constante" if usar_cte else "no constante"
        origen = {1: "automatico", 2: "forzado", 3: "forzado"}[modo]
        print(f"\nConfiguracion:")
        print(f"  Nodo      : i = {idx},  x_{idx} = {x[idx]:.10g}")
        print(f"  Esquema   : {esq}")
        print(f"  Tipo de h : {tipo_h} ({origen})")
        if aviso: print(f"  Aviso     : {aviso}")

        if orden == 1:
            if usar_cte:
                hh = h_usar
                if esq == "adelante":
                    d = (-3*y[idx] + 4*y[idx+1] - y[idx+2]) / (2*hh)
                    nodos_str = f"x{idx}, x{idx+1}, x{idx+2}"
                    formula_str = f"[-3 f(x{idx}) + 4 f(x{idx+1}) - f(x{idx+2})] / (2h)"
                elif esq == "central":
                    d = (y[idx+1] - y[idx-1]) / (2*hh)
                    nodos_str = f"x{idx-1}, x{idx}, x{idx+1}"
                    formula_str = f"[f(x{idx+1}) - f(x{idx-1})] / (2h)"
                else:
                    d = (3*y[idx] - 4*y[idx-1] + y[idx-2]) / (2*hh)
                    nodos_str = f"x{idx-2}, x{idx-1}, x{idx}"
                    formula_str = f"[3 f(x{idx}) - 4 f(x{idx-1}) + f(x{idx-2})] / (2h)"
            else:
                if esq == "adelante":
                    h1 = x[idx+1]-x[idx]; h2 = x[idx+2]-x[idx+1]
                    c0 = -(2*h1+h2)/(h1*(h1+h2)); c1 = (h1+h2)/(h1*h2); c2 = -h1/(h2*(h1+h2))
                    d = c0*y[idx] + c1*y[idx+1] + c2*y[idx+2]
                    nodos_str = f"x{idx}, x{idx+1}, x{idx+2}"
                    formula_str = "Lagrange 3 puntos (h variable), adelante"
                elif esq == "central":
                    h1 = x[idx]-x[idx-1]; h2 = x[idx+1]-x[idx]
                    ci = -h2/(h1*(h1+h2)); c0 = (h2-h1)/(h1*h2); cd = h1/(h2*(h1+h2))
                    d = ci*y[idx-1] + c0*y[idx] + cd*y[idx+1]
                    nodos_str = f"x{idx-1}, x{idx}, x{idx+1}"
                    formula_str = "Lagrange 3 puntos (h variable), central"
                else:
                    h1 = x[idx]-x[idx-1]; h2 = x[idx-1]-x[idx-2]
                    c0 = (2*h1+h2)/(h1*(h1+h2)); c1 = -(h1+h2)/(h1*h2); c2 = h1/(h2*(h1+h2))
                    d = c0*y[idx] + c1*y[idx-1] + c2*y[idx-2]
                    nodos_str = f"x{idx-2}, x{idx-1}, x{idx}"
                    formula_str = "Lagrange 3 puntos (h variable), atras"
        else:  # orden 2
            if usar_cte:
                hh = h_usar
                if esq == "adelante":
                    d = (y[idx] - 2*y[idx+1] + y[idx+2]) / hh**2
                    nodos_str = f"x{idx}, x{idx+1}, x{idx+2}"
                    formula_str = f"[f(x{idx}) - 2 f(x{idx+1}) + f(x{idx+2})] / h^2"
                elif esq == "central":
                    d = (y[idx-1] - 2*y[idx] + y[idx+1]) / hh**2
                    nodos_str = f"x{idx-1}, x{idx}, x{idx+1}"
                    formula_str = f"[f(x{idx-1}) - 2 f(x{idx}) + f(x{idx+1})] / h^2"
                else:
                    d = (y[idx] - 2*y[idx-1] + y[idx-2]) / hh**2
                    nodos_str = f"x{idx-2}, x{idx-1}, x{idx}"
                    formula_str = f"[f(x{idx}) - 2 f(x{idx-1}) + f(x{idx-2})] / h^2"
            else:
                if esq == "adelante":
                    h1 = x[idx+1]-x[idx]; h2 = x[idx+2]-x[idx+1]
                    d = 2*(y[idx]/(h1*(h1+h2)) - y[idx+1]/(h1*h2) + y[idx+2]/(h2*(h1+h2)))
                    nodos_str = f"x{idx}, x{idx+1}, x{idx+2}"
                    formula_str = "2a der. Lagrange 3 puntos (h variable), adelante"
                elif esq == "central":
                    h1 = x[idx]-x[idx-1]; h2 = x[idx+1]-x[idx]
                    d = 2*(y[idx-1]/(h1*(h1+h2)) - y[idx]/(h1*h2) + y[idx+1]/(h2*(h1+h2)))
                    nodos_str = f"x{idx-1}, x{idx}, x{idx+1}"
                    formula_str = "2a der. Lagrange 3 puntos (h variable), central"
                else:
                    h1 = x[idx-1]-x[idx-2]; h2 = x[idx]-x[idx-1]
                    d = 2*(y[idx-2]/(h1*(h1+h2)) - y[idx-1]/(h1*h2) + y[idx]/(h2*(h1+h2)))
                    nodos_str = f"x{idx-2}, x{idx-1}, x{idx}"
                    formula_str = "2a der. Lagrange 3 puntos (h variable), atras"

        x_eval = x[idx]
        if usar_cte:
            print(f"  h = {h_usar:.10g}")

    d_lag = float(dp_ref(x_eval))
    print(f"\nNodos usados : {nodos_str}")
    print(f"Formula      : {etiqueta} ≈ {formula_str}")

    print(f"\nResultado:")
    print(f"  {etiqueta}({x_eval:.10g}) ≈ {float(d):.10g}")
    print(f"  {etiqueta}_Lag({x_eval:.10g})  = {d_lag:.10g}")
    print(f"  |Error|             = {abs(float(d) - d_lag):.6e}")


def _modulo_derivacion_puntos():
    print("\n=== DERIVACION DESDE PUNTOS ===\n")
    print("  1. Primera derivada")
    print("  2. Segunda derivada")
    print("  3. Tercera derivada  (h constante, minimo 5 puntos)")
    print("  4. Cuarta derivada   (h constante, minimo 5 puntos)")
    sub = _pedir_int("\nElige [1-4]: ", condicion=lambda v: 1 <= v <= 4, error="elige entre 1 y 4")

    if sub in (1, 2):
        print("\nModo de trabajo:")
        print("  1. Aproximar en todos los nodos usando tabla (x, y)")
        print("  2. Aproximar en un nodo concreto usando tabla (x, y)  (adelante / central / atras)")
        print("  3. Aproximar en un punto desde la funcion f(x)        (adelante / central / atras)")
        modo_in = _pedir_int("Elige [1-3] (Enter=1): ", condicion=lambda v: 1 <= v <= 3,
                             error="elige entre 1 y 3", permitir_vacio=True, default=1)
        if modo_in == 2:
            _aproximar_derivada_punto_tabla(sub)
            return
        if modo_in == 3:
            _aproximar_derivada_punto(sub)
            return

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
        esq = _pedir_esquema()
        dp = np.polyder(p_lag)
        print(f"\nDerivada del polinomio global:")
        print(f"  P_{x.size-1}'(x) = {_fmt_poly(dp)}")

        if modo == 4:
            x_c, y_c, h_usar = _interpolar_uniforme(x, y)
            print(f"\n  Malla uniforme ({x_c.size} puntos, h = {h_usar:.10g}):")
            print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
            print("  " + "-" * 40)
            for i, (xi, yi) in enumerate(zip(x_c, y_c)):
                print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")
            print(f"\nConfiguracion elegida:")
            print(f"  Tipo de h : constante (interpolacion local Lagrange)")
            print(f"  h = {h_usar:.10g},  Esquema : {esq}")
            res = _d1_hcte(x_c, y_c, esq, h_usar)
        else:
            usar_cte, h_prom, aviso = _resolver_modo_h(es_cte, modo, h)
            tipo_h = "constante" if usar_cte else "no constante"
            origen = {1: "automatico", 2: "forzado", 3: "forzado"}[modo]
            print(f"\nConfiguracion elegida:")
            print(f"  Tipo de h : {tipo_h} ({origen})")
            print(f"  Esquema   : {esq}")
            if aviso: print(f"  Aviso     : {aviso}")
            h_usar = h0 if (usar_cte and es_cte) else h_prom
            res = _d1_hcte(x, y, esq, h_usar) if usar_cte else _d1_hvar(x, y, esq)

        if not res:
            print("No hay suficientes nodos para el esquema elegido."); return
        _print_d1(res, dp)

    elif sub == 2:
        modo = _pedir_modo_h()
        d2p = np.polyder(p_lag, 2)
        print(f"\nSegunda derivada del polinomio global:")
        print(f"  P_{x.size-1}''(x) = {_fmt_poly(d2p)}")

        if modo == 4:
            x_c, y_c, h_usar = _interpolar_uniforme(x, y)
            print(f"\n  Malla uniforme ({x_c.size} puntos, h = {h_usar:.10g}):")
            print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
            print("  " + "-" * 40)
            for i, (xi, yi) in enumerate(zip(x_c, y_c)):
                print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")
            print(f"\nConfiguracion elegida:")
            print(f"  Tipo de h : constante (interpolacion local Lagrange)")
            print(f"  h = {h_usar:.10g}")
            res = _d2_hcte(x_c, y_c, h_usar)
        else:
            usar_cte, h_prom, aviso = _resolver_modo_h(es_cte, modo, h)
            tipo_h = "constante" if usar_cte else "no constante"
            origen = {1: "automatico", 2: "forzado", 3: "forzado"}[modo]
            print(f"\nConfiguracion elegida:")
            print(f"  Tipo de h : {tipo_h} ({origen})")
            if aviso: print(f"  Aviso     : {aviso}")
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
    c = _pedir_float_array("coeficientes = ", min_len=1)
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


# ── Helpers de integracion desde funcion ─────────────────────────────────────

def _nodos_vals(f, a, b, n):
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    fs = [f(x) for x in xs]
    return h, xs, fs


def _print_nodos(xs, fs):
    print(f"\n  {'i':>4}  {'xi':>14}  {'f(xi)':>16}")
    print("  " + "─" * 38)
    for i, (xi, fi) in enumerate(zip(xs, fs)):
        print(f"  {i:>4}  {xi:>14.8g}  {fi:>16.10g}")


def _print_nodos_pesos(xs, fs, pesos):
    print(f"\n  {'i':>4}  {'xi':>14}  {'f(xi)':>16}  {'peso':>10}  {'peso*f(xi)':>16}")
    print("  " + "-" * 70)
    for i, (xi, fi, pi) in enumerate(zip(xs, fs, pesos)):
        print(f"  {i:>4}  {xi:>14.8g}  {fi:>16.10g}  {pi:>10.6g}  {pi*fi:>16.10g}")
    print("  " + "-" * 70)
    print(f"  {'Suma ponderada':>50}  {sum(pi*fi for pi, fi in zip(pesos, fs)):>16.10g}")


def _integ_trapecio_simple(f, a, b):
    fa, fb = f(a), f(b)
    res = (b - a) / 2 * (fa + fb)
    print(f"\n  f(a) = f({a:g}) = {fa:g}")
    print(f"  f(b) = f({b:g}) = {fb:g}")
    print(f"\n  I ≈ (b-a)/2 · [f(a) + f(b)] = {res:.10g}")
    return res


def _integ_simpson13_simple(f, a, b):
    h = (b - a) / 2
    x1 = a + h
    f0, f1, f2 = f(a), f(x1), f(b)
    res = (b - a) / 6 * (f0 + 4 * f1 + f2)
    print(f"\n  h = (b-a)/2 = {h:g}")
    print(f"  x0={a:g}  x1={x1:g}  x2={b:g}")
    print(f"  f0={f0:g}  f1={f1:g}  f2={f2:g}")
    print(f"\n  I ≈ (b-a)/6 · [f0 + 4f1 + f2] = {res:.10g}")
    return res


def _integ_simpson38_simple(f, a, b):
    h = (b - a) / 3
    x1, x2 = a + h, a + 2 * h
    f0, f1, f2, f3 = f(a), f(x1), f(x2), f(b)
    res = 3 * h / 8 * (f0 + 3 * f1 + 3 * f2 + f3)
    print(f"\n  h = (b-a)/3 = {h:g}")
    print(f"  x0={a:g}  x1={x1:g}  x2={x2:g}  x3={b:g}")
    print(f"  f0={f0:g}  f1={f1:g}  f2={f2:g}  f3={f3:g}")
    print(f"\n  I ≈ 3h/8 · [f0 + 3f1 + 3f2 + f3] = {res:.10g}")
    return res


def _integ_trapecio_comp_f(f, a, b, n):
    h, xs, fs = _nodos_vals(f, a, b, n)
    pesos = [2.0] * (n + 1); pesos[0] = pesos[-1] = 1.0
    _print_nodos_pesos(xs, fs, pesos)
    suma = sum(p * fi for p, fi in zip(pesos, fs))
    res = h / 2 * suma
    print(f"\n  I ≈ h/2 · [f0 + 2f1 + ... + 2f_{{n-1}} + fn]")
    print(f"    h={h:g}   suma_ponderada={suma:g}   I={res:.10g}")
    return res


def _integ_simpson13_comp_f(f, a, b, n):
    if n % 2 != 0:
        raise ValueError(f"n={n} debe ser par.")
    h, xs, fs = _nodos_vals(f, a, b, n)
    pesos = [1.0] * (n + 1)
    for i in range(1, n):
        pesos[i] = 4.0 if i % 2 != 0 else 2.0
    _print_nodos_pesos(xs, fs, pesos)
    suma = sum(p * fi for p, fi in zip(pesos, fs))
    res = h / 3 * suma
    print(f"\n  I ≈ h/3 · [f0 + 4f1 + 2f2 + 4f3 + ... + fn]")
    print(f"    h={h:g}   suma_ponderada={suma:g}   I={res:.10g}")
    return res


def _integ_simpson38_comp_f(f, a, b, n):
    if n % 3 != 0:
        raise ValueError(f"n={n} debe ser multiplo de 3.")
    h, xs, fs = _nodos_vals(f, a, b, n)
    pesos = [1.0] * (n + 1)
    for i in range(1, n):
        pesos[i] = 2.0 if i % 3 == 0 else 3.0
    _print_nodos_pesos(xs, fs, pesos)
    suma = sum(p * fi for p, fi in zip(pesos, fs))
    res = 3 * h / 8 * suma
    print(f"\n  I ≈ 3h/8 · [f0 + 3f1 + 3f2 + 2f3 + ... + fn]")
    print(f"    h={h:g}   suma_ponderada={suma:g}   I={res:.10g}")
    return res


def _integ_punto_medio(f, a, b):
    xm = (a + b) / 2
    fm = f(xm)
    res = (b - a) * fm
    print(f"\n  x_m = (a+b)/2 = {xm:g}")
    print(f"  f(x_m) = {fm:g}")
    print(f"\n  I ≈ (b-a) · f((a+b)/2) = {res:.10g}")
    return res


def _integ_dos_puntos(f, a, b):
    h = (b - a) / 3
    x1, x2 = a + h, a + 2 * h
    f1, f2 = f(x1), f(x2)
    res = 3 * h / 2 * (f1 + f2)
    print(f"\n  h = (b-a)/3 = {h:g}")
    print(f"  x1={x1:g}  x2={x2:g}")
    print(f"  f(x1)={f1:g}  f(x2)={f2:g}")
    print(f"\n  I ≈ 3h/2 · [f(x1) + f(x2)] = {res:.10g}")
    return res


def _abiertas_tabular(x, y):
    """
    Formulas abiertas de Newton-Cotes desde datos tabulados.
    x, y son SOLO los puntos interiores (sin los extremos del intervalo).
    Se admiten 1, 2, 3 o 4 puntos interiores, igualmente espaciados.
    Devuelve (resultado, nombre, formula, h, a, b).
    """
    n = x.size
    if n < 1 or n > 4:
        raise ValueError(f"Formulas abiertas admiten 1–4 puntos interiores ({n} dados).")
    if n > 1:
        h_arr = np.diff(x)
        h0 = float(h_arr[0])
        if not np.allclose(h_arr, h0, atol=1e-10, rtol=1e-8):
            raise ValueError("Los puntos interiores deben estar igualmente espaciados.")
        h = h0
    else:
        raise ValueError("Con 1 punto interior debes indicar h manualmente (usa modo 7).")
    a = float(x[0]) - h
    b = float(x[-1]) + h
    if n == 2:
        res = 3*h/2 * (y[0] + y[1])
        nombre = "Dos puntos (abierta)"
        formula = "3h/2 · (f₁ + f₂)"
    elif n == 3:
        res = 4*h/3 * (2*y[0] - y[1] + 2*y[2])
        nombre = "Milne — 3 puntos interiores"
        formula = "4h/3 · (2f₁ - f₂ + 2f₃)"
    else:  # n == 4
        res = 5*h/24 * (11*y[0] + y[1] + y[2] + 11*y[3])
        nombre = "4 puntos interiores"
        formula = "5h/24 · (11f₁ + f₂ + f₃ + 11f₄)"
    return float(res), nombre, formula, h, a, b


_INTEG_COMP = {"4", "5", "6"}

_INTEG_OPS = {
    "1": "Trapecio simple            [cerrada, 2 pts]",
    "2": "Simpson 1/3 simple         [cerrada, 3 pts]",
    "3": "Simpson 3/8 simple         [cerrada, 4 pts]",
    "4": "Trapecio compuesto         [cerrada, n+1 pts]",
    "5": "Simpson 1/3 compuesto      [cerrada, n par]",
    "6": "Simpson 3/8 compuesto      [cerrada, n mult. 3]",
    "7": "Punto medio                [abierta, 1 pt interior]",
    "8": "Dos puntos                 [abierta, 2 pts interiores]",
    "9": "Desde datos tabulados      [Trapecio / S1/3 / S3/8 / abiertas]",
}


def calcular_integracion_funcion(f, a, b, op, n=None):
    """Calcula una formula de integracion desde f(x) y devuelve (resultado, metodo, n_graf)."""
    if op == "1":
        return _integ_trapecio_simple(f, a, b), "Trapecio simple", 2
    if op == "2":
        return _integ_simpson13_simple(f, a, b), "Simpson 1/3 simple", 2
    if op == "3":
        return _integ_simpson38_simple(f, a, b), "Simpson 3/8 simple", 3
    if op == "4":
        if n is None:
            raise ValueError("Trapecio compuesto requiere n.")
        return _integ_trapecio_comp_f(f, a, b, n), "Trapecio compuesto", n
    if op == "5":
        if n is None:
            raise ValueError("Simpson 1/3 compuesto requiere n.")
        return _integ_simpson13_comp_f(f, a, b, n), "Simpson 1/3 compuesto", n
    if op == "6":
        if n is None:
            raise ValueError("Simpson 3/8 compuesto requiere n.")
        return _integ_simpson38_comp_f(f, a, b, n), "Simpson 3/8 compuesto", n
    if op == "7":
        return _integ_punto_medio(f, a, b), "Punto medio", 1
    if op == "8":
        return _integ_dos_puntos(f, a, b), "Dos puntos", 2
    raise ValueError("Metodo de integracion no soportado.")


def menu_graficar_integracion():
    if not _MPLOK:
        print("  matplotlib no disponible."); return
    print("\n=== GRAFICAR / COMPARAR — INTEGRACION ===\n")
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return
    print("\nMetodos a comparar (varios con espacio):")
    for k in ("1", "2", "3", "4", "5", "6", "7", "8"):
        print(f"  {k}. {_INTEG_OPS[k]}")
    try:
        raw = _input("Metodos: ").strip().split()
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    sels = [s for s in raw if s in set("12345678")]
    if not sels:
        sels = ["1", "2", "3"]

    n_por_metodo = {}
    for s in sels:
        if s in _INTEG_COMP:
            try:
                n = _pedir_int(f"n para {_INTEG_OPS[s]} = ", condicion=lambda v: v > 0, error="n debe ser > 0")
            except VolverAtras:
                print(f"  n invalido para metodo {s}; se omite.")
                continue
            n_por_metodo[s] = n

    resultados = []
    for s in sels:
        try:
            res, metodo, n_graf = calcular_integracion_funcion(f, a, b, s, n_por_metodo.get(s))
        except Exception as e:
            print(f"  Error metodo {s}: {e}")
            continue
        resultados.append({'metodo': metodo, 'resultado': res, 'n': n_graf})

    if not resultados:
        print("No hubo resultados para graficar."); return
    if len(resultados) == 1:
        d = resultados[0]
        _gr.integracion(f, a, b, d['n'], d['metodo'], d['resultado'])
    else:
        _gr.integracion_comparacion(f, a, b, resultados, f_str=f_str)


def _modulo_integracion():
    print("\n=== INTEGRACION NUMERICA (Newton-Cotes) ===\n")
    print("  Formulas cerradas — simples:")
    for k in ("1", "2", "3"):
        print(f"    {k}. {_INTEG_OPS[k]}")
    print("  Formulas cerradas — compuestas (desde f(x)):")
    for k in ("4", "5", "6"):
        print(f"    {k}. {_INTEG_OPS[k]}")
    print("  Formulas abiertas:")
    for k in ("7", "8"):
        print(f"    {k}. {_INTEG_OPS[k]}")
    print("  Datos tabulados:")
    print(f"    9. {_INTEG_OPS['9']}")

    op = _pedir_texto("\nElige [1-9]: ")
    if op not in _INTEG_OPS:
        print("Opcion invalida"); return

    # ── Modo datos tabulados ────────────────────────────────────────────────
    if op == "9":
        print("\n  Formulas cerradas (incluyen extremos):")
        print("    1. Trapecio compuesto  (h variable)")
        print("    2. Simpson 1/3         (n par, h cte)")
        print("    3. Simpson 3/8         (n mult.3, h cte)")
        print("  Formulas abiertas (solo puntos interiores, sin extremos):")
        print("    4. Dos puntos          (2 pts interiores, h cte)")
        print("    5. Milne               (3 pts interiores, h cte)")
        print("    6. Cuatro puntos       (4 pts interiores, h cte)")
        sub = _pedir_int("  Metodo [1-6]: ", condicion=lambda v: 1 <= v <= 6, error="elige entre 1 y 6")

        # ── Formulas abiertas desde datos ──────────────────────────────────
        if sub in (4, 5, 6):
            n_req = {4: 2, 5: 3, 6: 4}[sub]
            print(f"\nIngresa los {n_req} punto(s) INTERIOR(ES) del intervalo.")
            print("  (NO incluyas los extremos a y b del intervalo de integracion)")
            try:
                x = _leer_arr("x (interiores)"); y = _leer_arr("y (interiores)")
            except ValueError as e:
                print(e); return
            if x.size != y.size:
                print("Error: x e y deben tener igual longitud."); return
            if x.size != n_req:
                print(f"Error: se necesitan exactamente {n_req} punto(s) interior(es) ({x.size} dados)."); return
            orden = np.argsort(x); x = x[orden]; y = y[orden]
            try:
                r, nombre, formula, h, a, b = _abiertas_tabular(x, y)
            except ValueError as e:
                print(f"Error: {e}"); return
            print(f"\nMetodo     : {nombre}")
            print(f"Formula    : I ≈ {formula}")
            print(f"Intervalo  : [{a:.6g}, {b:.6g}]  (inferido de los puntos interiores)")
            print(f"h = {h:.6g}")
            print(f"\nPuntos interiores usados:")
            print(f"  {'i':>4} {'x_i':>14} {'y_i':>16}")
            print("  " + "-" * 38)
            for i, (xi, yi) in enumerate(zip(x, y), 1):
                print(f"  {i:>4} {xi:>14.8g} {yi:>16.10g}")
            print(f"\n{'═'*52}")
            print(f"  RESULTADO: I ≈ {r:.10g}")
            print(f"{'═'*52}")
            return

        # ── Formulas cerradas desde datos ───────────────────────────────────
        print("\nIngresa los puntos tabulares (incluyendo extremos).")
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
        print(f"\n{n_sub} subintervalos  a={x[0]:.6g}  b={x[-1]:.6g}")
        try:
            if sub == 1:
                r = _trapecio(x, y)
                print(f"\nTrapecio compuesto:  I ≈ {r:.10g}")
            elif sub == 2:
                r = _simpson13(x, y)
                print(f"\nSimpson 1/3:  I ≈ {r:.10g}")
            else:
                r = _simpson38(x, y)
                print(f"\nSimpson 3/8:  I ≈ {r:.10g}")
        except ValueError as e:
            print(f"Error: {e}")
        return

    # ── Modo desde funcion ──────────────────────────────────────────────────
    f, f_str = _pedir_funcion()
    if f is None: return
    print("\nIntervalo [a, b]:")
    a, b = _pedir_intervalo()
    if a is None: return

    n = None
    if op in _INTEG_COMP:
        try:
            n = _pedir_int("Numero de subintervalos n: ", condicion=lambda v: v >= 1, error="n debe ser entero positivo")
        except VolverAtras:
            return

    print(f"\nIntegrando f(x) = {f_str}  en  [{a:g}, {b:g}]")
    try:
        res, nombre_op, n_graf = calcular_integracion_funcion(f, a, b, op, n)
    except ValueError as e:
        print(f"Error: {e}"); return

    print(f"\n{'═'*52}")
    print(f"  RESULTADO: I ≈ {res:.10g}")
    print(f"{'═'*52}")

    exact_s = _input("\nValor exacto (Enter para omitir): ").strip()
    if exact_s:
        try:
            exact = float(exact_s)
            err_abs = abs(res - exact)
            err_rel = err_abs / abs(exact) * 100 if abs(exact) > 1e-15 else float("nan")
            print(f"  Error absoluto: {err_abs:.6e}")
            print(f"  Error relativo: {err_rel:.4f}%")
        except Exception:
            pass
    _preguntar_grafica({'tipo': 'integracion', 'f': f, 'a': a, 'b': b,
                        'n': n_graf, 'metodo': nombre_op, 'resultado': res})


def menu_derivacion_integracion():
    while True:
        print("\n=== DERIVACION E INTEGRACION NUMERICA (TEMA 4) ===\n")
        print("  1. Derivacion desde puntos  (1a, 2a, 3a, 4a derivada)")
        print("  2. Derivacion desde polinomio ingresado")
        print("  3. Integracion numerica     (Newton-Cotes: simples, compuestas, abiertas)")
        print("  0. Volver al menu principal")
        try:
            op = _input("\nElige [0-3]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op == "0":
            break
        try:
            op = int(op)
        except Exception:
            print("Opcion invalida"); continue
        try:
            if op == 1:   _modulo_derivacion_puntos()
            elif op == 2: _modulo_polinomio()
            elif op == 3: _modulo_integracion()
            else:         print("Opcion invalida")
        except VolverAtras:
            continue




__all__ = [name for name in globals() if not name.startswith("__")]
