"""tema_interpolacion.py — Interpolacion y aproximacion."""

from common import *

# ═══════════════════════════════════════════════════════════════════════════════
# TEMA 3: INTERPOLACION Y APROXIMACION
# ═══════════════════════════════════════════════════════════════════════════════

def _ingresar_datos_xy():
    print("\nFormato: valores separados por espacios  (ej: 1 2 3 4 5)\n")
    _avisar_esc()
    while True:
        x = _pedir_float_array("x = ", min_len=2)
        y = _pedir_float_array("y = ", min_len=2)
        if len(x) != len(y):
            print(f"Error: x tiene {len(x)} valores e y tiene {len(y)}.")
            continue
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


def _print_tabla_reg_lineal(x, y):
    x2 = x**2; xy = x * y
    print("\nTabla auxiliar — regresion lineal:")
    print(f"  {'i':>4} {'x':>12} {'y':>12} {'x^2':>14} {'x*y':>14}")
    print("  " + "-" * 62)
    for i, vals in enumerate(zip(x, y, x2, xy)):
        print(f"  {i:>4} {vals[0]:>12.6g} {vals[1]:>12.6g} {vals[2]:>14.6g} {vals[3]:>14.6g}")
    print("  " + "-" * 62)
    print(f"  {'Suma':>4} {np.sum(x):>12.6g} {np.sum(y):>12.6g} {np.sum(x2):>14.6g} {np.sum(xy):>14.6g}")
    print("\nSistema normal:")
    print(f"  {len(x):.6g}*b0 + {np.sum(x):.6g}*b1 = {np.sum(y):.6g}")
    print(f"  {np.sum(x):.6g}*b0 + {np.sum(x2):.6g}*b1 = {np.sum(xy):.6g}")


def _print_tabla_reg_polinomial(x, y, grado):
    max_pow = 2 * grado
    potencias = [x**p for p in range(1, max_pow + 1)]
    productos = [(x**p) * y for p in range(1, grado + 1)]
    print(f"\nTabla auxiliar — regresion polinomial grado {grado}:")
    headers = ["i", "x", "y"] + [f"x^{p}" for p in range(2, max_pow + 1)] + [f"x^{p}*y" for p in range(1, grado + 1)]
    print("  " + " ".join(f"{h:>12}" for h in headers))
    print("  " + "-" * (13 * len(headers)))
    for i in range(len(x)):
        vals = [i, x[i], y[i]] + [potencias[p-1][i] for p in range(2, max_pow + 1)] + [productos[p-1][i] for p in range(1, grado + 1)]
        print("  " + " ".join(f"{v:>12.6g}" for v in vals))
    sums = ["Suma", np.sum(x), np.sum(y)] + [np.sum(potencias[p-1]) for p in range(2, max_pow + 1)] + [np.sum(productos[p-1]) for p in range(1, grado + 1)]
    print("  " + "-" * (13 * len(headers)))
    print("  " + " ".join(f"{v:>12.6g}" if not isinstance(v, str) else f"{v:>12}" for v in sums))
    print("\nSistema normal:")
    for fila in range(grado + 1):
        lhs = []
        for col in range(grado + 1):
            power = fila + col
            coef = len(x) if power == 0 else np.sum(x**power)
            lhs.append(f"{coef:.6g}*b{col}")
        rhs = np.sum((x**fila) * y)
        print(f"  {' + '.join(lhs)} = {rhs:.6g}")


def _print_tabla_reg_exponencial(x, y):
    ly = np.log(y); x2 = x**2; xly = x * ly
    print("\nTabla auxiliar — regresion exponencial linealizada:")
    print("  y = b*exp(a*x)  ->  ln(y) = ln(b) + a*x")
    print(f"  {'i':>4} {'x':>12} {'y':>12} {'ln(y)':>14} {'x^2':>14} {'x*ln(y)':>14}")
    print("  " + "-" * 78)
    for i, vals in enumerate(zip(x, y, ly, x2, xly)):
        print(f"  {i:>4} {vals[0]:>12.6g} {vals[1]:>12.6g} {vals[2]:>14.6g} {vals[3]:>14.6g} {vals[4]:>14.6g}")
    print("  " + "-" * 78)
    print(f"  {'Suma':>4} {np.sum(x):>12.6g} {np.sum(y):>12.6g} {np.sum(ly):>14.6g} {np.sum(x2):>14.6g} {np.sum(xly):>14.6g}")


def _print_tabla_reg_multiple(xv, yv, zv):
    x2 = xv**2; y2 = yv**2; xy = xv*yv; xz = xv*zv; yz = yv*zv
    print("\nTabla auxiliar — regresion multiple z = b0 + b1*x + b2*y:")
    print(f"  {'i':>4} {'x':>10} {'y':>10} {'z':>10} {'x^2':>12} {'y^2':>12} {'x*y':>12} {'x*z':>12} {'y*z':>12}")
    print("  " + "-" * 100)
    for i, vals in enumerate(zip(xv, yv, zv, x2, y2, xy, xz, yz)):
        print(f"  {i:>4} " + " ".join(f"{v:>10.6g}" for v in vals[:3]) + " " + " ".join(f"{v:>12.6g}" for v in vals[3:]))
    print("  " + "-" * 100)
    print(f"  {'Suma':>4} {np.sum(xv):>10.6g} {np.sum(yv):>10.6g} {np.sum(zv):>10.6g} {np.sum(x2):>12.6g} {np.sum(y2):>12.6g} {np.sum(xy):>12.6g} {np.sum(xz):>12.6g} {np.sum(yz):>12.6g}")
    print("\nSistema normal:")
    print(f"  {len(xv):.6g}*b0 + {np.sum(xv):.6g}*b1 + {np.sum(yv):.6g}*b2 = {np.sum(zv):.6g}")
    print(f"  {np.sum(xv):.6g}*b0 + {np.sum(x2):.6g}*b1 + {np.sum(xy):.6g}*b2 = {np.sum(xz):.6g}")
    print(f"  {np.sum(yv):.6g}*b0 + {np.sum(xy):.6g}*b1 + {np.sum(y2):.6g}*b2 = {np.sum(yz):.6g}")


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


def _print_lagrange_bases(x, y):
    print("\nPolinomios base de Lagrange:")
    for i in range(len(x)):
        factores = []
        den = 1.0
        for j in range(len(x)):
            if i == j:
                continue
            factores.append(f"(x - {x[j]:.6g})")
            den *= (x[i] - x[j])
        print(f"  L{i}(x) = {'*'.join(factores)} / ({den:.6g})")
        print(f"  Termino: y{i}*L{i}(x) = {y[i]:.6g}*L{i}(x)")


def _eval_poli(coefs, xp):
    return np.polyval(np.asarray(coefs)[::-1], np.asarray(xp, dtype=float))


def menu_graficar_interp():
    if not _MPLOK:
        print("  matplotlib no disponible."); return
    print("\n=== GRAFICAR — INTERPOLACION / REGRESION ===\n")
    print("Ingresa los puntos de datos (x, y):")
    x, y = _ingresar_datos_xy()
    if x is None: return
    print("\nMetodos a graficar (varios con espacio):")
    print("  1. Reg. lineal      2. Reg. polinomial    3. Newton    4. Lagrange")
    try:
        raw = _input("Metodos: ").strip().split()
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    sels = [r for r in raw if r in ("1", "2", "3", "4")]
    if not sels: sels = ["1"]
    curvas = []
    x_fine = np.linspace(float(x.min()), float(x.max()), 400)
    for s in sels:
        try:
            if s == "1":
                b0, b1, _ = regresion_lineal(x, y)
                curvas.append({'xs': x_fine, 'ys': b0 + b1 * x_fine, 'label': 'Reg. lineal'})
            elif s == "2":
                m = _pedir_int("Grado m para reg. polinomial (Enter=2): ",
                               condicion=lambda v: v >= 1,
                               error="grado debe ser >= 1",
                               permitir_vacio=True, default=2)
                cs, _ = regresion_polinomial(x, y, m)
                curvas.append({'xs': x_fine, 'ys': np.array([float(_eval_poli(cs, xi)) for xi in x_fine]),
                               'label': f'Reg. polin. grado {m}'})
            elif s == "3":
                if len(np.unique(x)) < len(x): print("Newton requiere xi distintos."); continue
                y_vals = np.array([float(interpolacion_newton(x, y, xi)[0]) for xi in x_fine])
                curvas.append({'xs': x_fine, 'ys': y_vals, 'label': 'Newton'})
            elif s == "4":
                if len(np.unique(x)) < len(x): print("Lagrange requiere xi distintos."); continue
                y_vals = np.array([float(interpolacion_lagrange(x, y, xi)) for xi in x_fine])
                curvas.append({'xs': x_fine, 'ys': y_vals, 'label': 'Lagrange'})
        except Exception as e:
            print(f"  Error metodo {s}: {e}"); continue
    if curvas:
        _gr.interpolacion(x, y, curvas, titulo="Interpolacion / Regresion")


def menu_interp_aprox():
    while True:
        print("\n=== INTERPOLACION Y APROXIMACION (TEMA 3) ===\n")
        print("  1. Regresion lineal            y = b0 + b1*x")
        print("  2. Regresion polinomial        y = b0 + b1*x + ... + bm*x^m")
        print("  3. Regresion funcion conocida  F(x) = a0*phi0(x) + ...")
        print("  4. Regresion exponencial       y = b*exp(a*x)")
        print("  5. Regresion multiple 3D       z = b0 + b1*x + b2*y")
        print("  6. Interpolacion de Newton     (diferencias divididas)")
        print("  7. Interpolacion de Lagrange")
        print("  8. Graficar")
        print("  0. Volver al menu principal")
        try:
            op_s = _input("\nElige [0-8]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op_s == "0":
            break
        if op_s == "8":
            menu_graficar_interp(); continue
        try:
            op = int(op_s)
        except Exception:
            print("Opcion invalida"); continue
        if op not in range(1, 8):
            print("Opcion invalida"); continue

        if op == 5:
            print("\nIngresa los puntos (x, y, z). Minimo 3 puntos.\n")
            _avisar_esc()
            while True:
                xv = _pedir_float_array("x = ", min_len=3)
                yv = _pedir_float_array("y = ", min_len=3)
                zv = _pedir_float_array("z = ", min_len=3)
                if not (len(xv) == len(yv) == len(zv)):
                    print("Error: x, y, z deben tener la misma longitud.")
                    continue
                break
            try:
                coefs, ec = regresion_multiple(xv, yv, zv)
            except Exception as e:
                print(f"Error: {e}"); continue
            _print_tabla_reg_multiple(xv, yv, zv)
            _print_resultado(
                "REGRESION MULTIPLE 3D\n  z = b0 + b1*x + b2*y",
                [("b0", coefs[0]), ("b1", coefs[1]), ("b2", coefs[2])], ec
            )
            ev = _input("\nEvaluar en (x y) (Enter para omitir): ").strip()
            if ev:
                while True:
                    try:
                        xp, yp = [float(v) for v in ev.split()]
                        print(f"  F({xp}, {yp}) = {coefs[0]+coefs[1]*xp+coefs[2]*yp:.10g}")
                        break
                    except Exception:
                        print("Error: ingresa dos numeros separados por espacio.")
                        ev = _input("Evaluar en (x y) (Enter para omitir): ").strip()
                        if not ev:
                            break
            continue

        x, y = _ingresar_datos_xy()
        if x is None: continue
        print("\nPuntos ingresados:")
        _print_tabla_xy(x, y)

        if op == 1:
            print("\n--- Regresion lineal ---")
            try:
                b0, b1, ec = regresion_lineal(x, y)
            except ValueError as e:
                print(f"Error: {e}"); continue
            _print_tabla_reg_lineal(x, y)
            _print_resultado("REGRESION LINEAL   y = b0 + b1*x", [("b0", b0), ("b1", b1)], ec)
            ev = _input("\nEvaluar en x (Enter para omitir): ").strip()
            if ev:
                while True:
                    try:
                        xp = float(ev); print(f"  F({xp}) = {b0+b1*xp:.10g}"); break
                    except Exception:
                        print("Error: numero invalido.")
                        ev = _input("Evaluar en x (Enter para omitir): ").strip()
                        if not ev: break
            _preguntar_grafica({'tipo': 'interpolacion', 'x_datos': x, 'y_datos': y,
                                'f_ajuste': lambda xi, _b0=b0, _b1=b1: _b0 + _b1 * xi,
                                'label': f'y = {b0:+.4g} + {b1:+.4g}x', 'titulo': 'Regresion lineal'})

        elif op == 2:
            print("\n--- Regresion polinomial ---")
            try:
                m = _pedir_int("Grado m = ", condicion=lambda v: v >= 1, error="entero >= 1 requerido")
            except VolverAtras:
                continue
            try:
                coefs, ec = regresion_polinomial(x, y, m)
            except ValueError as e:
                print(f"Error: {e}"); continue
            _print_tabla_reg_polinomial(x, y, m)
            _print_resultado(f"REGRESION POLINOMIAL  grado {m}",
                             [(f"b{k}", coefs[k]) for k in range(len(coefs))], ec)
            ev = _input("\nEvaluar en x (Enter para omitir): ").strip()
            if ev:
                while True:
                    try:
                        xp = float(ev); print(f"  P({xp}) = {float(_eval_poli(coefs, xp)):.10g}"); break
                    except Exception:
                        print("Error: numero invalido.")
                        ev = _input("Evaluar en x (Enter para omitir): ").strip()
                        if not ev: break
            _coefs_g = coefs.copy()
            _preguntar_grafica({'tipo': 'interpolacion', 'x_datos': x, 'y_datos': y,
                                'f_ajuste': lambda xi, _c=_coefs_g: float(_eval_poli(_c, xi)),
                                'label': f'Reg. polin. grado {m}', 'titulo': f'Regresion polinomial grado {m}'})

        elif op == 3:
            print("\n--- Regresion funcion conocida ---")
            print(AYUDA_FUNCIONES)
            print("\nEjemplos de funciones base: 1  |  x  |  x^2  |  sin(x)  |  exp(x)\n")
            try:
                r = _pedir_int("Numero de funciones base r = ", condicion=lambda v: v >= 1, error="entero >= 1 requerido")
            except VolverAtras:
                continue
            funcs = []; nombres = []
            for k in range(r):
                while True:
                    phi_s = _pedir_texto(f"phi{k}(x) = ")
                    try:
                        phi_e = crear_funcion_segura(phi_s); phi_e(1)
                        phi_n = np.vectorize(phi_e); phi_n(np.array([1.0]))
                        break
                    except Exception as e:
                        print(f"Error phi{k}: {e}")
                funcs.append(phi_n); nombres.append(phi_s)
            else:
                try:
                    coefs, ec = regresion_funcion_conocida(x, y, funcs)
                except ValueError as e:
                    print(f"Error: {e}"); continue
                modelo = " + ".join(f"a{k}*({nb})" for k, nb in enumerate(nombres))
                _print_resultado(f"REGRESION FUNCION CONOCIDA\n  F(x) = {modelo}",
                                 [(f"a{k}", coefs[k]) for k in range(len(coefs))], ec)
                ev = _input("\nEvaluar en x (Enter para omitir): ").strip()
                if ev:
                    while True:
                        try:
                            xp = float(ev); xa = np.array([xp])
                            yp = sum(c * phi(xa) for c, phi in zip(coefs, funcs))
                            print(f"  F({xp}) = {np.asarray(yp).ravel()[0]:.10g}")
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            ev = _input("Evaluar en x (Enter para omitir): ").strip()
                            if not ev: break

        elif op == 4:
            print("\n--- Regresion exponencial ---")
            try:
                a, b_e, ec = regresion_exponencial(x, y)
            except ValueError as e:
                print(f"Error: {e}"); continue
            _print_tabla_reg_exponencial(x, y)
            _print_resultado("REGRESION EXPONENCIAL   y = b*exp(a*x)", [("a", a), ("b", b_e)], ec)
            ev = _input("\nEvaluar en x (Enter para omitir): ").strip()
            if ev:
                while True:
                    try:
                        xp = float(ev); print(f"  F({xp}) = {b_e*math.exp(a*xp):.10g}"); break
                    except Exception:
                        print("Error: numero invalido.")
                        ev = _input("Evaluar en x (Enter para omitir): ").strip()
                        if not ev: break
            _preguntar_grafica({'tipo': 'interpolacion', 'x_datos': x, 'y_datos': y,
                                'f_ajuste': lambda xi, _a=a, _b=b_e: _b * math.exp(_a * xi),
                                'label': f'y = {b_e:.4g}·exp({a:.4g}x)', 'titulo': 'Regresion exponencial'})

        elif op == 6:
            if len(np.unique(x)) < len(x):
                print("Error: xi deben ser distintos para interpolacion."); continue
            print("\n--- Interpolacion de Newton (diferencias divididas) ---")
            try:
                ver_tabla = _input("Mostrar tabla de diferencias divididas? (S/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError, VolverAtras):
                ver_tabla = "n"
            if ver_tabla in ("", "s", "si", "sí", "y", "yes", "1"):
                _print_tabla_dif(x, y)
            try:
                xp = _pedir_float("\nEvaluar en x = ")
            except VolverAtras:
                continue
            y_e, _, _ = interpolacion_newton(x, y, xp)
            print(f"\nP({xp}) = {float(y_e):.10g}")
            _preguntar_grafica({'tipo': 'interpolacion', 'x_datos': x, 'y_datos': y,
                                'f_ajuste': lambda xi, _x=x, _y=y: float(interpolacion_newton(_x, _y, xi)[0]),
                                'label': 'Newton', 'titulo': 'Interpolacion de Newton'})

        elif op == 7:
            if len(np.unique(x)) < len(x):
                print("Error: xi deben ser distintos para interpolacion."); continue
            print("\n--- Interpolacion de Lagrange ---")
            try:
                ver_bases = _input("Mostrar polinomios base L_i(x)? (S/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError, VolverAtras):
                ver_bases = "n"
            if ver_bases in ("", "s", "si", "sí", "y", "yes", "1"):
                _print_lagrange_bases(x, y)
            try:
                xp = _pedir_float("Evaluar en x = ")
            except VolverAtras:
                continue
            y_e = interpolacion_lagrange(x, y, xp)
            print(f"\nP({xp}) = {float(y_e):.10g}")
            _preguntar_grafica({'tipo': 'interpolacion', 'x_datos': x, 'y_datos': y,
                                'f_ajuste': lambda xi, _x=x, _y=y: float(interpolacion_lagrange(_x, _y, xi)),
                                'label': 'Lagrange', 'titulo': 'Interpolacion de Lagrange'})




__all__ = [name for name in globals() if not name.startswith("__")]
