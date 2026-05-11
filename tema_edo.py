"""tema_edo.py — Ecuaciones diferenciales ordinarias."""

from common import *

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
    _avisar_esc()
    while True:
        f_str = _pedir_texto("f(x,y) = ")
        try:
            f = crear_funcion_xy_segura(f_str); f(0, 1)
        except Exception as e:
            print(f"Error: {e}")
            continue
        return f, f_str


def _pedir_ci():
    while True:
        x0 = _pedir_float("x0 = ")
        y0 = _pedir_float("y0 = ")
        xf = _pedir_float("x final = ")
        if xf <= x0:
            print("Error: x final debe ser > x0")
            continue
        h_str = _input("Paso h (Enter para especificar n): ").strip()
        if h_str:
            try:
                h = float(h_str)
            except Exception:
                print("Error: valor invalido.")
                continue
            if h <= 0:
                print("Error: h debe ser > 0")
                continue
        else:
            n = _pedir_int("Numero de pasos n = ", condicion=lambda v: v > 0, error="n debe ser > 0")
            h = (xf - x0) / n
        return x0, y0, xf, h


def _edo_header(cols):
    print("\n" + f"{'n':<6} " + " ".join(f"{c:<20}" for c in cols))
    print("-" * (6 + 21 * len(cols)))


def _validar_paso(h):
    if h <= 0:
        raise ValueError("h debe ser > 0.")


class _ResultadoEDO(tuple):
    """Resultado desempacable (y, xs, ys) que tambien se comporta como escalar."""

    def __new__(cls, valor, xs, ys):
        return super().__new__(cls, (valor, xs, ys))

    @property
    def valor(self):
        return self[0]

    @property
    def xs(self):
        return self[1]

    @property
    def ys(self):
        return self[2]

    def __float__(self):
        return float(self.valor)

    def _op(self, other, fn):
        return fn(float(self), other)

    def _rop(self, other, fn):
        return fn(other, float(self))

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._rop(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._rop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._rop(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._rop(other, lambda a, b: a / b)


def metodo_euler(f, x0, y0, xf, h):
    _validar_paso(h)
    print("\n--- Euler:  y_{n+1} = y_n + h*f(x_n, y_n) ---")
    _edo_header(["x_n", "y_n", "f(x_n,y_n)", "y_{n+1}"])
    xs_arr = [x0]; ys_arr = [y0]
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try: k = f(x, y)
        except Exception as e: print(f"Error: {e}"); return None, np.array(xs_arr), np.array(ys_arr)
        yn = y + h_ef * k
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
        xs_arr.append(x); ys_arr.append(y)
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return _ResultadoEDO(y, np.array(xs_arr), np.array(ys_arr))


def metodo_rk2_pm(f, x0, y0, xf, h):
    _validar_paso(h)
    print("\n--- RK2 Punto Medio:  k1=f(x,y), k2=f(x+h/2, y+h/2*k1), y_{n+1}=y+h*k2 ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "y_{n+1}"])
    xs_arr = [x0]; ys_arr = [y0]
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x, y); k2 = f(x + h_ef/2, y + h_ef/2 * k1)
        except Exception as e: print(f"Error: {e}"); return None, np.array(xs_arr), np.array(ys_arr)
        yn = y + h_ef * k2
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
        xs_arr.append(x); ys_arr.append(y)
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return _ResultadoEDO(y, np.array(xs_arr), np.array(ys_arr))


def metodo_rk2_heun(f, x0, y0, xf, h):
    _validar_paso(h)
    print("\n--- RK2 Heun:  k1=f(x,y), k2=f(x+h, y+h*k1), y_{n+1}=y+(h/2)*(k1+k2) ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "y_{n+1}"])
    xs_arr = [x0]; ys_arr = [y0]
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x, y); k2 = f(x + h_ef, y + h_ef * k1)
        except Exception as e: print(f"Error: {e}"); return None, np.array(xs_arr), np.array(ys_arr)
        yn = y + (h_ef / 2) * (k1 + k2)
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
        xs_arr.append(x); ys_arr.append(y)
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return _ResultadoEDO(y, np.array(xs_arr), np.array(ys_arr))


def metodo_rk4(f, x0, y0, xf, h):
    _validar_paso(h)
    print("\n--- RK4:  y_{n+1} = y_n + (h/6)*(k1 + 2k2 + 2k3 + k4) ---")
    _edo_header(["x_n", "y_n", "k1", "k2", "k3", "k4", "y_{n+1}"])
    xs_arr = [x0]; ys_arr = [y0]
    x, y = x0, y0; paso = 0
    while x < xf - 1e-14:
        h_ef = min(h, xf - x)
        try:
            k1 = f(x,            y)
            k2 = f(x + h_ef/2,   y + h_ef/2 * k1)
            k3 = f(x + h_ef/2,   y + h_ef/2 * k2)
            k4 = f(x + h_ef,     y + h_ef   * k3)
        except Exception as e: print(f"Error: {e}"); return None, np.array(xs_arr), np.array(ys_arr)
        yn = y + (h_ef / 6) * (k1 + 2*k2 + 2*k3 + k4)
        print(f"{paso:<6} {x:<20.10g} {y:<20.10g} {k1:<20.10g} {k2:<20.10g} {k3:<20.10g} {k4:<20.10g} {yn:<20.10g}")
        x = round(x + h_ef, 14); y = yn; paso += 1
        xs_arr.append(x); ys_arr.append(y)
    print(f"\ny(x={xf}) ≈ {y:.10g}")
    return _ResultadoEDO(y, np.array(xs_arr), np.array(ys_arr))


# ─── Sistema de 2 EDOs ────────────────────────────────────────────────────────

_AYUDA_TXY = (
    "Funciones disponibles (variables: t, x, y):\n"
    "  sin, cos, tan, exp, log (=ln), sqrt, abs\n"
    "Constantes: pi, e  |  Potencias: usa ^ o **\n"
    "Ejemplos:  0.5*x - 0.02*x*y   |   -0.3*y + 0.01*x*y"
)


def _crear_func_txy(expr_str):
    evaluar = crear_evaluador_seguro(expr_str, ("t", "x", "y"))
    def f(t, x, y):
        return evaluar(t=t, x=x, y=y)
    return f


def _pedir_sistema2():
    print(_AYUDA_TXY)
    print("\nIngresa el sistema:  dx/dt = f1(t,x,y)   dy/dt = f2(t,x,y)\n")
    _avisar_esc()
    while True:
        f1_str = _pedir_texto("f1(t,x,y) = ")
        f2_str = _pedir_texto("f2(t,x,y) = ")
        try:
            f1 = _crear_func_txy(f1_str); f1(0, 1, 1)
            f2 = _crear_func_txy(f2_str); f2(0, 1, 1)
        except Exception as e:
            print(f"Error: {e}")
            continue
        return f1, f2, f1_str, f2_str


def _pedir_ci_sistema():
    while True:
        t0 = _pedir_float("t0 = ")
        x0 = _pedir_float("x(t0) = ")
        y0 = _pedir_float("y(t0) = ")
        tf = _pedir_float("t final = ")
        if tf <= t0:
            print("Error: t final > t0")
            continue
        h_str = _input("Paso h (Enter para n): ").strip()
        if h_str:
            try:
                h = float(h_str)
            except Exception:
                print("Error: valor invalido.")
                continue
        else:
            n = _pedir_int("n = ", condicion=lambda v: v > 0, error="n debe ser > 0")
            h = (tf - t0) / n
        if h <= 0:
            print("Error: h debe ser > 0")
            continue
        return t0, x0, y0, tf, h


def _s2_euler_paso(f1, f2, t, x, y, h):
    k1 = f1(t, x, y); k2 = f2(t, x, y)
    return x + h*k1, y + h*k2


def _s2_rk2pm_paso(f1, f2, t, x, y, h):
    k1x = f1(t, x, y); k1y = f2(t, x, y)
    k2x = f1(t+h/2, x+h/2*k1x, y+h/2*k1y)
    k2y = f2(t+h/2, x+h/2*k1x, y+h/2*k1y)
    return x + h*k2x, y + h*k2y


def _s2_heun_paso(f1, f2, t, x, y, h):
    k1x = f1(t, x, y); k1y = f2(t, x, y)
    k2x = f1(t+h, x+h*k1x, y+h*k1y)
    k2y = f2(t+h, x+h*k1x, y+h*k1y)
    return x + (h/2)*(k1x+k2x), y + (h/2)*(k1y+k2y)


def _s2_rk4_paso(f1, f2, t, x, y, h):
    k1x = f1(t,      x,          y         )
    k1y = f2(t,      x,          y         )
    k2x = f1(t+h/2,  x+h/2*k1x, y+h/2*k1y)
    k2y = f2(t+h/2,  x+h/2*k1x, y+h/2*k1y)
    k3x = f1(t+h/2,  x+h/2*k2x, y+h/2*k2y)
    k3y = f2(t+h/2,  x+h/2*k2x, y+h/2*k2y)
    k4x = f1(t+h,    x+h*k3x,   y+h*k3y  )
    k4y = f2(t+h,    x+h*k3x,   y+h*k3y  )
    return (x + h/6*(k1x+2*k2x+2*k3x+k4x),
            y + h/6*(k1y+2*k2y+2*k3y+k4y))


def _s2_paso_detalle(f1, f2, paso_fn, t, x, y, h):
    if paso_fn is _s2_euler_paso:
        k1x = f1(t, x, y); k1y = f2(t, x, y)
        xn, yn = x + h*k1x, y + h*k1y
        return xn, yn, [("k1x", k1x), ("k1y", k1y)]
    if paso_fn is _s2_rk2pm_paso:
        k1x = f1(t, x, y); k1y = f2(t, x, y)
        k2x = f1(t+h/2, x+h/2*k1x, y+h/2*k1y)
        k2y = f2(t+h/2, x+h/2*k1x, y+h/2*k1y)
        xn, yn = x + h*k2x, y + h*k2y
        return xn, yn, [("k1x", k1x), ("k1y", k1y), ("k2x", k2x), ("k2y", k2y)]
    if paso_fn is _s2_heun_paso:
        k1x = f1(t, x, y); k1y = f2(t, x, y)
        k2x = f1(t+h, x+h*k1x, y+h*k1y)
        k2y = f2(t+h, x+h*k1x, y+h*k1y)
        xn, yn = x + (h/2)*(k1x+k2x), y + (h/2)*(k1y+k2y)
        return xn, yn, [("k1x", k1x), ("k1y", k1y), ("k2x", k2x), ("k2y", k2y)]
    if paso_fn is _s2_rk4_paso:
        k1x = f1(t,      x,          y         )
        k1y = f2(t,      x,          y         )
        k2x = f1(t+h/2,  x+h/2*k1x, y+h/2*k1y)
        k2y = f2(t+h/2,  x+h/2*k1x, y+h/2*k1y)
        k3x = f1(t+h/2,  x+h/2*k2x, y+h/2*k2y)
        k3y = f2(t+h/2,  x+h/2*k2x, y+h/2*k2y)
        k4x = f1(t+h,    x+h*k3x,   y+h*k3y  )
        k4y = f2(t+h,    x+h*k3x,   y+h*k3y  )
        xn = x + h/6*(k1x+2*k2x+2*k3x+k4x)
        yn = y + h/6*(k1y+2*k2y+2*k3y+k4y)
        return xn, yn, [
            ("k1x", k1x), ("k1y", k1y), ("k2x", k2x), ("k2y", k2y),
            ("k3x", k3x), ("k3y", k3y), ("k4x", k4x), ("k4y", k4y),
        ]
    xn, yn = paso_fn(f1, f2, t, x, y, h)
    return xn, yn, []


def _sistema2_run(f1, f2, f1_str, f2_str, paso_fn, nombre, t0, x0, y0, tf, h):
    _validar_paso(h)
    print(f"\n--- Sistema 2 EDOs — {nombre} ---")
    print(f"dx/dt = {f1_str}")
    print(f"dy/dt = {f2_str}")
    hdr = f"{'n':<5} {'t_n':<12} {'x_n':<14} {'y_n':<14} {'x_n+1':<14} {'y_n+1':<14}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    t_arr = [t0]; x_arr = [x0]; y_arr = [y0]
    t, x, y = t0, x0, y0; paso = 0
    while t < tf - 1e-14:
        h_ef = min(h, tf - t)
        try:
            xn, yn, detalle = _s2_paso_detalle(f1, f2, paso_fn, t, x, y, h_ef)
        except Exception as e:
            print(f"Error: {e}"); return np.array(t_arr), np.array(x_arr), np.array(y_arr)
        print(f"{paso:<5} {t:<12.6g} {x:<14.8g} {y:<14.8g} {xn:<14.8g} {yn:<14.8g}")
        if detalle:
            print("      " + "  ".join(f"{k}={v:.8g}" for k, v in detalle))
        t = round(t + h_ef, 14); x, y = xn, yn; paso += 1
        t_arr.append(t); x_arr.append(x); y_arr.append(y)
    print(f"\nx(t={tf}) ≈ {x:.10g}")
    print(f"y(t={tf}) ≈ {y:.10g}")
    return np.array(t_arr), np.array(x_arr), np.array(y_arr)


# ─── EDO de 2° orden ──────────────────────────────────────────────────────────

_AYUDA_TXV = (
    "Funciones disponibles (variables: t, x, v  donde v = x'):\n"
    "  sin, cos, tan, exp, log (=ln), sqrt, abs\n"
    "Constantes: pi, e  |  Potencias: usa ^ o **\n"
    "Ejemplos:\n"
    "  x'' + 4x = 0           ->  -4*x\n"
    "  x'' + 0.5x' + 4x = 0  ->  -0.5*v - 4*x\n"
    "  x'' + 2x' + 5x = sin(t)->  sin(t) - 2*v - 5*x"
)


def _crear_func_txv(expr_str):
    evaluar = crear_evaluador_seguro(expr_str, ("t", "x", "v"))
    def f(t, x, v):
        return evaluar(t=t, x=x, v=v)
    return f


def _pedir_edo2():
    print(_AYUDA_TXV)
    print("\nIngresa  x'' = f(t, x, v):\n")
    _avisar_esc()
    while True:
        f_str = _pedir_texto("f(t, x, v) = ")
        try:
            f = _crear_func_txv(f_str); f(0, 1, 0)
        except Exception as e:
            print(f"Error: {e}")
            continue
        return f, f_str


def _pedir_ci_edo2():
    while True:
        t0 = _pedir_float("t0 = ")
        x0 = _pedir_float("x(t0) = ")
        v0 = _pedir_float("x'(t0) = ")
        tf = _pedir_float("t final = ")
        if tf <= t0:
            print("Error: t final > t0")
            continue
        h_str = _input("Paso h (Enter para n): ").strip()
        if h_str:
            try:
                h = float(h_str)
            except Exception:
                print("Error: valor invalido.")
                continue
        else:
            n = _pedir_int("n = ", condicion=lambda v: v > 0, error="n debe ser > 0")
            h = (tf - t0) / n
        if h <= 0:
            print("Error: h debe ser > 0")
            continue
        return t0, x0, v0, tf, h


def _e2_euler_paso(f, t, x, v, h):
    return x + h*v, v + h*f(t, x, v)


def _e2_rk2pm_paso(f, t, x, v, h):
    k1x = v;           k1v = f(t,      x,          v         )
    k2x = v + h/2*k1v; k2v = f(t+h/2,  x+h/2*k1x, v+h/2*k1v)
    return x + h*k2x, v + h*k2v


def _e2_rk4_paso(f, t, x, v, h):
    k1x = v;           k1v = f(t,      x,          v         )
    k2x = v + h/2*k1v; k2v = f(t+h/2,  x+h/2*k1x, v+h/2*k1v)
    k3x = v + h/2*k2v; k3v = f(t+h/2,  x+h/2*k2x, v+h/2*k2v)
    k4x = v + h*k3v;   k4v = f(t+h,    x+h*k3x,   v+h*k3v  )
    return (x + h/6*(k1x+2*k2x+2*k3x+k4x),
            v + h/6*(k1v+2*k2v+2*k3v+k4v))


def _edo2_run(f, f_str, paso_fn, nombre, t0, x0, v0, tf, h):
    _validar_paso(h)
    print(f"\n--- EDO 2° orden — {nombre}  (v = x') ---")
    print(f"x'' = {f_str}   =>   x' = v,   v' = {f_str}")
    hdr = f"{'n':<5} {'t_n':<12} {'x_n':<14} {'v_n':<14} {'x_n+1':<14} {'v_n+1':<14}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    t_arr = [t0]; x_arr = [x0]; v_arr = [v0]
    t, x, v = t0, x0, v0; paso = 0
    while t < tf - 1e-14:
        h_ef = min(h, tf - t)
        try:
            xn, vn = paso_fn(f, t, x, v, h_ef)
        except Exception as e:
            print(f"Error: {e}"); return np.array(t_arr), np.array(x_arr), np.array(v_arr)
        print(f"{paso:<5} {t:<12.6g} {x:<14.8g} {v:<14.8g} {xn:<14.8g} {vn:<14.8g}")
        t = round(t + h_ef, 14); x, v = xn, vn; paso += 1
        t_arr.append(t); x_arr.append(x); v_arr.append(v)
    print(f"\nx(t={tf})  ≈ {x:.10g}")
    print(f"x'(t={tf}) ≈ {v:.10g}")
    return np.array(t_arr), np.array(x_arr), np.array(v_arr)


# ─── Comparación de los 4 métodos sobre la misma EDO escalar ──────────────────

def comparacion_metodos_edo(f, x0, y0, xf, h):
    print(f"\n=== COMPARACION — y({xf}) con h={h} ===")
    resultados = []
    datos_list = []
    for nombre, fn in [("Euler", metodo_euler), ("RK2 Punto Medio", metodo_rk2_pm),
                        ("Heun", metodo_rk2_heun), ("RK4", metodo_rk4)]:
        val, xs_arr, ys_arr = fn(f, x0, y0, xf, h)
        resultados.append((nombre, val))
        datos_list.append({'xs': xs_arr, 'ys': ys_arr, 'label': nombre})
    print(f"\n{'='*50}")
    print(f"TABLA COMPARATIVA — y(x={xf})")
    print(f"{'='*50}")
    print(f"{'Metodo':<20} {'Resultado':<20}")
    print("-" * 42)
    for nombre, val in resultados:
        s = f"{val:.10g}" if val is not None else "Error"
        print(f"{nombre:<20} {s:<20}")
    return datos_list


# ─── Seleccion de ejes para graficas EDO ─────────────────────────────────────

def _pedir_ejes_sistema():
    print("\nEjes a graficar:")
    print("  1. x(t)            2. y(t)            3. x(t) e y(t)")
    print("  4. Plano de fase  x vs y")
    e = _pedir_int("Elige [1-4] (Enter=3): ", condicion=lambda v: 1 <= v <= 4,
                   error="elige entre 1 y 4", permitir_vacio=True, default=3)
    return {1: 'xt', 2: 'yt', 3: 'ambas', 4: 'xy'}.get(e, 'ambas')


def _pedir_ejes_2orden():
    print("\nEjes a graficar:")
    print("  1. x(t)            2. x'(t)           3. x(t) y x'(t)")
    print("  4. Plano de fase  x vs x'")
    e = _pedir_int("Elige [1-4] (Enter=3): ", condicion=lambda v: 1 <= v <= 4,
                   error="elige entre 1 y 4", permitir_vacio=True, default=3)
    return {1: 'xt', 2: 'vt', 3: 'ambas', 4: 'xv'}.get(e, 'ambas')


# ─── Menu graficar EDOs ───────────────────────────────────────────────────────

def menu_graficar_odes():
    if not _MPLOK:
        print("  matplotlib no disponible."); return
    print("\n=== GRAFICAR — EDOs ===\n")
    print("  1. EDO escalar  y' = f(x, y)")
    print("  2. Sistema 2 EDOs")
    print("  3. EDO 2° orden  x'' = f(t, x, x')")
    print("  0. Volver")
    try:
        tipo = _input("\nElige [0-3]: ").strip()
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    if tipo == "0": return

    if tipo == "1":
        f, f_str = _pedir_ode()
        if f is None: return
        ci = _pedir_ci()
        if ci is None: return
        x0, y0, xf, h = ci
        print("\nMetodos a graficar (varios con espacio):")
        print("  1. Euler   2. RK2 PM   3. Heun   4. RK4")
        try:
            raw = _input("Metodos: ").strip().split()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            return
        sels = [r for r in raw if r in ("1", "2", "3", "4")]
        if not sels: sels = ["1", "2", "3", "4"]
        mapa = {"1": (metodo_euler, "Euler"), "2": (metodo_rk2_pm, "RK2 PM"),
                "3": (metodo_rk2_heun, "Heun"), "4": (metodo_rk4, "RK4")}
        datos_list = []
        for s in sels:
            fn, nm = mapa[s]
            _, xs, ys = fn(f, x0, y0, xf, h)
            datos_list.append({'xs': xs, 'ys': ys, 'label': nm})
        _gr.edo_escalar(datos_list, eje_x='x', eje_y='y(x)',
                        titulo=f"y' = {f_str}   y({x0})={y0}")

    elif tipo == "2":
        res = _pedir_sistema2()
        if res[0] is None: return
        f1, f2, f1_str, f2_str = res
        ci = _pedir_ci_sistema()
        if ci is None: return
        t0, x0, y0, tf, h = ci
        eje = _pedir_ejes_sistema()
        print("\nMetodos a graficar (varios con espacio):")
        print("  1. Euler   2. RK2 PM   3. Heun   4. RK4")
        try:
            raw = _input("Metodos: ").strip().split()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            return
        sels = [r for r in raw if r in ("1", "2", "3", "4")]
        if not sels: sels = ["1", "2", "3", "4"]
        mapa = {"1": (_s2_euler_paso, "Euler"), "2": (_s2_rk2pm_paso, "RK2 PM"),
                "3": (_s2_heun_paso, "Heun"),   "4": (_s2_rk4_paso, "RK4")}
        datos_list = []
        for s in sels:
            paso_fn, nm = mapa[s]
            t_a, x_a, y_a = _sistema2_run(f1, f2, f1_str, f2_str, paso_fn, nm, t0, x0, y0, tf, h)
            datos_list.append({'t': t_a, 'x': x_a, 'y': y_a, 'label': nm})
        _gr.edo_sistema(datos_list, eje=eje,
                        titulo=f"dx/dt={f1_str}   dy/dt={f2_str}")

    elif tipo == "3":
        f, f_str = _pedir_edo2()
        if f is None: return
        ci = _pedir_ci_edo2()
        if ci is None: return
        t0, x0, v0, tf, h = ci
        eje = _pedir_ejes_2orden()
        print("\nMetodos a graficar (varios con espacio):")
        print("  1. Euler   2. RK2 PM   3. RK4")
        try:
            raw = _input("Metodos: ").strip().split()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            return
        sels = [r for r in raw if r in ("1", "2", "3")]
        if not sels: sels = ["1", "2", "3"]
        mapa = {"1": (_e2_euler_paso, "Euler"), "2": (_e2_rk2pm_paso, "RK2 PM"),
                "3": (_e2_rk4_paso, "RK4")}
        datos_list = []
        for s in sels:
            paso_fn, nm = mapa[s]
            t_a, x_a, v_a = _edo2_run(f, f_str, paso_fn, nm, t0, x0, v0, tf, h)
            datos_list.append({'t': t_a, 'x': x_a, 'v': v_a, 'label': nm})
        _gr.edo_2orden(datos_list, eje=eje,
                       titulo=f"x'' = {f_str}   x({t0})={x0}  x'({t0})={v0}")


# ─── Menu EDOs ────────────────────────────────────────────────────────────────

def menu_odes():
    while True:
        print("\n=== ECUACIONES DIFERENCIALES ORDINARIAS (TEMA 5) ===\n")
        print("  EDO escalar  y' = f(x, y):")
        print("   1. Euler")
        print("   2. RK2 Punto Medio")
        print("   3. RK2 Heun  (Trapecio)")
        print("   4. RK4  (Runge-Kutta orden 4)")
        print()
        print("  Sistema 2 EDOs  dx/dt=f1(t,x,y)  dy/dt=f2(t,x,y):")
        print("   5. Euler")
        print("   6. RK2 Punto Medio")
        print("   7. Heun")
        print("   8. RK4")
        print()
        print("  EDO 2° orden  x'' = f(t, x, x'):")
        print("   9. Euler")
        print("  10. RK2 Punto Medio")
        print("  11. RK4")
        print()
        print("  12. Comparacion  Euler / RK2pm / Heun / RK4  (EDO escalar)")
        print("  13. Graficar")
        print()
        print("   0. Volver al menu principal")
        try:
            op = _input("\nElige [0-13]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op == "0":
            break
        if op == "13":
            menu_graficar_odes(); continue
        try:
            op = int(op)
        except Exception:
            print("Opcion invalida"); continue

        try:
            _menu_odes_op(op)
        except VolverAtras:
            continue


def _menu_odes_op(op):
        if op in (1, 2, 3, 4, 12):
            f, f_str = _pedir_ode()
            if f is None: return
            ci = _pedir_ci()
            if ci is None: return
            x0, y0, xf, h = ci
            print(f"\ny' = {f_str},  y({x0}) = {y0},  x ∈ [{x0}, {xf}],  h = {h}")
            if op == 1:
                val, xs, ys = metodo_euler(f, x0, y0, xf, h)
                _preguntar_grafica({'tipo': 'edo', 'xs': xs, 'ys': ys,
                                    'metodo': 'Euler', 'titulo': f"Euler  y'={f_str}"})
            elif op == 2:
                val, xs, ys = metodo_rk2_pm(f, x0, y0, xf, h)
                _preguntar_grafica({'tipo': 'edo', 'xs': xs, 'ys': ys,
                                    'metodo': 'RK2 PM', 'titulo': f"RK2 PM  y'={f_str}"})
            elif op == 3:
                val, xs, ys = metodo_rk2_heun(f, x0, y0, xf, h)
                _preguntar_grafica({'tipo': 'edo', 'xs': xs, 'ys': ys,
                                    'metodo': 'Heun', 'titulo': f"Heun  y'={f_str}"})
            elif op == 4:
                val, xs, ys = metodo_rk4(f, x0, y0, xf, h)
                _preguntar_grafica({'tipo': 'edo', 'xs': xs, 'ys': ys,
                                    'metodo': 'RK4', 'titulo': f"RK4  y'={f_str}"})
            elif op == 12:
                datos_list = comparacion_metodos_edo(f, x0, y0, xf, h)
                if _MPLOK:
                    try:
                        resp = _input("\n¿Graficar comparacion? (s/n): ").strip().lower()
                    except (KeyboardInterrupt, EOFError, VolverAtras):
                        resp = 'n'
                    if resp in ('s', 'si', 'sí', 'y'):
                        _gr.edo_escalar(datos_list, titulo=f"Comparacion  y'={f_str}")

        elif op in (5, 6, 7, 8):
            res = _pedir_sistema2()
            if res[0] is None: return
            f1, f2, f1_str, f2_str = res
            ci = _pedir_ci_sistema()
            if ci is None: return
            t0, x0, y0, tf, h = ci
            paso_map = {5: (_s2_euler_paso,  "Euler"),
                        6: (_s2_rk2pm_paso,  "RK2 Punto Medio"),
                        7: (_s2_heun_paso,   "Heun"),
                        8: (_s2_rk4_paso,    "RK4")}
            paso_fn, nombre = paso_map[op]
            t_a, x_a, y_a = _sistema2_run(f1, f2, f1_str, f2_str, paso_fn, nombre, t0, x0, y0, tf, h)
            if _MPLOK:
                try:
                    resp = _input("\n¿Graficar? (s/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError, VolverAtras):
                    resp = 'n'
                if resp in ('s', 'si', 'sí', 'y'):
                    eje = _pedir_ejes_sistema()
                    _gr.edo_sistema([{'t': t_a, 'x': x_a, 'y': y_a, 'label': nombre}],
                                    eje=eje, titulo=f"Sistema — {nombre}")

        elif op in (9, 10, 11):
            f, f_str = _pedir_edo2()
            if f is None: return
            ci = _pedir_ci_edo2()
            if ci is None: return
            t0, x0, v0, tf, h = ci
            print(f"\nx'' = {f_str},  x({t0})={x0}, x'({t0})={v0}, t∈[{t0},{tf}], h={h}")
            paso_map = {9:  (_e2_euler_paso, "Euler"),
                        10: (_e2_rk2pm_paso, "RK2 Punto Medio"),
                        11: (_e2_rk4_paso,   "RK4")}
            paso_fn, nombre = paso_map[op]
            t_a, x_a, v_a = _edo2_run(f, f_str, paso_fn, nombre, t0, x0, v0, tf, h)
            if _MPLOK:
                try:
                    resp = _input("\n¿Graficar? (s/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError, VolverAtras):
                    resp = 'n'
                if resp in ('s', 'si', 'sí', 'y'):
                    eje = _pedir_ejes_2orden()
                    _gr.edo_2orden([{'t': t_a, 'x': x_a, 'v': v_a, 'label': nombre}],
                                   eje=eje, titulo=f"EDO 2° orden — {nombre}  x''={f_str}")

        else:
            print("Opcion invalida")




__all__ = [name for name in globals() if not name.startswith("__")]
