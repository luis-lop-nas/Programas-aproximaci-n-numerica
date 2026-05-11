"""tema_edp.py — Ecuaciones en derivadas parciales."""

from common import *

# ═══════════════════════════════════════════════════════════════════════════════
# MENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def _pedir_bc(nombre, var):
    while True:
        raw = _pedir_texto(f"  {nombre} (constante o expr. en {var}): ")
        try:
            c = float(raw)
            return lambda v, _c=c: _c
        except ValueError:
            try:
                evaluar = crear_evaluador_seguro(raw, (var,))
                def bc(v, _ev=evaluar, _v=var):
                    return _ev(**{_v: v})
                bc(0.0)
                return bc
            except Exception as e:
                print(f"  Error en '{raw}': {e}")
                continue


def _edp_datos():
    print("\nDominio rectangular [xa, xb] x [ya, yb]:")
    print("  Ejemplo: xa=0  xb=1  ya=0  yb=1  (cuadrado unitario)")
    print("           xa=0  xb=2  ya=0  yb=1  (rectangulo 2x1)\n")
    _avisar_esc()
    while True:
        xa = _pedir_float("  xa = ")
        xb = _pedir_float("  xb = ")
        ya = _pedir_float("  ya = ")
        yb = _pedir_float("  yb = ")
        if xb <= xa or yb <= ya:
            print("Error: xb > xa y yb > ya.")
            continue
        break
    print()
    print("Grid — elige como especificar el tamano:")
    print("  N  -> mismo numero de subdivisiones en x e y")
    print("  NM -> N subdivisiones en x y M en y (distintos)")
    print("  h  -> tamanyo de paso (calcula N automaticamente)")
    while True:
        modo = _input("Escribe N, NM o h: ").strip().lower()
        if modo == "h":
            h = _pedir_float("  h = ", condicion=lambda v: v > 0, error="h debe ser > 0")
            Nx = round((xb - xa) / h)
            Ny = round((yb - ya) / h)
            if Nx < 2 or Ny < 2:
                print("Error: dominio demasiado pequenyo para ese h")
                continue
            print(f"  -> Nx={Nx} (hx={(xb-xa)/Nx:.6g}),  Ny={Ny} (hy={(yb-ya)/Ny:.6g})")
            break
        elif modo == "nm":
            Nx = _pedir_int("  N (nodos en x) = ", condicion=lambda v: v >= 2, error="N debe ser >= 2")
            Ny = _pedir_int("  M (nodos en y) = ", condicion=lambda v: v >= 2, error="M debe ser >= 2")
            print(f"  -> hx={(xb-xa)/Nx:.6g},  hy={(yb-ya)/Ny:.6g}")
            break
        elif modo == "n":
            N = _pedir_int("  N = ", condicion=lambda v: v >= 2, error="N debe ser >= 2")
            Nx = Ny = N
            print(f"  -> hx={(xb-xa)/Nx:.6g},  hy={(yb-ya)/Ny:.6g}")
            break
        else:
            print("Error: escribe N, NM o h.")
    print("\nCondiciones de contorno Dirichlet:")
    print("  Escribe un numero (constante) o una expresion en la variable indicada.")
    print("  Ejemplos de constante : 0   100   -5.2")
    print("  Ejemplos de expresion : x   y^2   sin(x)   exp(y)   x*(1-x)\n")
    bc_bot   = _pedir_bc("u(x, ya) — inferior  ", "x")
    bc_top   = _pedir_bc("u(x, yb) — superior  ", "x")
    bc_left  = _pedir_bc("u(xa, y) — izquierda ", "y")
    bc_right = _pedir_bc("u(xb, y) — derecha   ", "y")
    if None in (bc_bot, bc_top, bc_left, bc_right):
        return None
    return xa, xb, ya, yb, Nx, Ny, bc_bot, bc_top, bc_left, bc_right


def _edp_solver(u, xs, ys, hx, hy, Nx, Ny=None, f_src=None, omega=1.0, tol=1e-5, maxit=10000, return_hist=False):
    if maxit is None:
        maxit = 10000
    # Compatibilidad con la firma antigua para mallas cuadradas:
    # _edp_solver(u, xs, ys, hx, hy, N, f_src, omega, tol, maxit)
    if callable(Ny):
        old_f_src, old_omega, old_tol, old_maxit = Ny, f_src, omega, tol
        Ny = Nx
        f_src = old_f_src
        omega = 1.0 if old_omega is None else old_omega
        tol = 1e-5 if old_tol is None else old_tol
        maxit = 10000 if old_maxit is None else old_maxit
    if Ny is None:
        Ny = Nx
    if f_src is None:
        f_src = lambda x, y: 0.0
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (hx2 + hy2)
    hist = []
    for it in range(1, maxit + 1):
        max_delta = 0.0
        for i in range(1, Nx):
            for j in range(1, Ny):
                u_gs = (hy2 * (u[i+1, j] + u[i-1, j])
                      + hx2 * (u[i, j+1] + u[i, j-1])
                      - hx2 * hy2 * f_src(xs[i], ys[j])) / denom
                u_new = (1.0 - omega) * u[i, j] + omega * u_gs
                delta = abs(u_new - u[i, j])
                if delta > max_delta:
                    max_delta = delta
                u[i, j] = u_new
        if return_hist:
            hist.append((it, max_delta))
        if max_delta < tol:
            if return_hist:
                return it, max_delta, hist
            return it, max_delta
    if return_hist:
        return maxit, max_delta, hist
    return maxit, max_delta


def _edp_print_convergencia(hist, max_filas=20):
    print("\nTabla de convergencia EDP:")
    print(f"  {'iter':>8} {'delta_max':>16}")
    print("  " + "-" * 27)
    if len(hist) <= max_filas:
        muestra = hist
    else:
        mitad = max_filas // 2
        muestra = hist[:mitad] + [(None, None)] + hist[-mitad:]
    for it, delta in muestra:
        if it is None:
            print(f"  {'...':>8} {'...':>16}")
        else:
            print(f"  {it:>8} {delta:>16.6e}")


def _edp_imprimir(u, xs, ys, Nx, Ny):
    if max(Nx, Ny) > 10:
        print(f"\n(Grid {Nx+1}x{Ny+1} — se muestran esquinas y centro)")
        pts_x = [0, Nx // 2, Nx]
        pts_y = [0, Ny // 2, Ny]
        print(f"\n{'y\\x':>10}", end="")
        for i in pts_x:
            print(f"  {xs[i]:>12.6g}", end="")
        print()
        for j in reversed(pts_y):
            print(f"{ys[j]:>10.6g}", end="")
            for i in pts_x:
                print(f"  {u[i, j]:>12.6g}", end="")
            print()
        return
    print(f"\n{'y\\x':>8}", end="")
    for i in range(Nx + 1):
        print(f"  {xs[i]:>10.4f}", end="")
    print()
    print("-" * (8 + 12 * (Nx + 1)))
    for j in range(Ny, -1, -1):
        print(f"{ys[j]:>8.4f}", end="")
        for i in range(Nx + 1):
            print(f"  {u[i, j]:>10.6f}", end="")
        print()


def _edp_run(es_poisson, usa_sor):
    datos = _edp_datos()
    if datos is None: return
    xa, xb, ya, yb, Nx, Ny, bc_bot, bc_top, bc_left, bc_right = datos
    if es_poisson:
        print("\nTermino fuente f(x,y) tal que  nabla^2 u = f(x,y):")
        print("  Variables: x, y — mismas funciones que en EDOs")
        print("  Ejemplos: -2   |  x^2 + y^2   |  sin(pi*x)*sin(pi*y)   |  -4\n")
        while True:
            raw = _pedir_texto("  f(x,y) = ")
            try:
                evaluar_src = crear_evaluador_seguro(raw, ("x", "y"))
                def f_src(x, y, _ev=evaluar_src): return _ev(x=x, y=y)
                f_src(0.0, 0.0)
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        def f_src(x, y): return 0.0
    if usa_sor:
        rho = 0.5 * (math.cos(math.pi / Nx) + math.cos(math.pi / Ny))
        omega_opt = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))
        print(f"\nFactor de sobre-relajacion omega (optimo aprox. {omega_opt:.4f}):")
        omega = _pedir_float(f"  omega (Enter = {omega_opt:.4f}): ",
                             permitir_vacio=True, default=omega_opt)
        if not (1.0 < omega < 2.0):
            print("  Aviso: omega debe estar en (1, 2) para convergencia garantizada.")
    else:
        omega = 1.0
    tol, maxit = _pedir_tol_iter(1e-5, 10000)
    xs = np.linspace(xa, xb, Nx + 1)
    ys = np.linspace(ya, yb, Ny + 1)
    hx = (xb - xa) / Nx
    hy = (yb - ya) / Ny
    u = np.zeros((Nx + 1, Ny + 1))
    for i in range(Nx + 1):
        u[i, 0] = bc_bot(xs[i])
        u[i, Ny] = bc_top(xs[i])
    for j in range(Ny + 1):
        u[0, j] = bc_left(ys[j])
        u[Nx, j] = bc_right(ys[j])
    nombre = "Poisson" if es_poisson else "Laplace"
    metodo = f"SOR (omega={omega:.4f})" if usa_sor else "Gauss-Seidel"
    print(f"\nResolviendo {nombre} — {metodo}  (grid {Nx+1}x{Ny+1})...")
    it, delta, hist = _edp_solver(u, xs, ys, hx, hy, Nx, Ny, f_src, omega, tol, maxit, return_hist=True)
    _edp_print_convergencia(hist)
    print(f"\n{'=' * 55}")
    if delta < tol:
        print(f"  Convergencia en {it} iteraciones  (delta = {delta:.3e})")
    else:
        print(f"  No convergio en {maxit} iteraciones  (delta = {delta:.3e})")
    print(f"{'=' * 55}")
    _edp_imprimir(u, xs, ys, Nx, Ny)
    _preguntar_grafica({'tipo': 'edp', 'u': u, 'xs': xs, 'ys': ys,
                        'titulo': f'{nombre} — {metodo}'})


def _edp_comparar_gs_sor():
    datos = _edp_datos()
    if datos is None: return
    xa, xb, ya, yb, Nx, Ny, bc_bot, bc_top, bc_left, bc_right = datos
    try:
        es_poisson = _input("\nPoisson con termino fuente? (s/N): ").strip().lower() in ("s", "si", "sí", "y", "yes", "1")
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    if es_poisson:
        while True:
            raw = _pedir_texto("  f(x,y) = ")
            try:
                evaluar_src = crear_evaluador_seguro(raw, ("x", "y"))
                def f_src(x, y, _ev=evaluar_src): return _ev(x=x, y=y)
                f_src(0.0, 0.0)
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        def f_src(x, y): return 0.0
    tol, maxit = _pedir_tol_iter(1e-5, 10000)
    xs = np.linspace(xa, xb, Nx + 1)
    ys = np.linspace(ya, yb, Ny + 1)
    hx = (xb - xa) / Nx
    hy = (yb - ya) / Ny

    def crear_u():
        u = np.zeros((Nx + 1, Ny + 1))
        for i in range(Nx + 1):
            u[i, 0] = bc_bot(xs[i]); u[i, Ny] = bc_top(xs[i])
        for j in range(Ny + 1):
            u[0, j] = bc_left(ys[j]); u[Nx, j] = bc_right(ys[j])
        return u

    rho = 0.5 * (math.cos(math.pi / Nx) + math.cos(math.pi / Ny))
    omega_opt = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))
    resultados = []
    for nombre, omega in [("Gauss-Seidel", 1.0), ("SOR", omega_opt)]:
        u = crear_u()
        it, delta, hist = _edp_solver(u, xs, ys, hx, hy, Nx, Ny, f_src, omega, tol, maxit, return_hist=True)
        resultados.append((nombre, omega, it, delta, hist))

    print("\nComparacion GS vs SOR:")
    print(f"  {'Metodo':<16} {'omega':>10} {'iter':>10} {'delta final':>16}")
    print("  " + "-" * 58)
    for nombre, omega, it, delta, _ in resultados:
        print(f"  {nombre:<16} {omega:>10.5f} {it:>10} {delta:>16.6e}")
    for nombre, _, _, _, hist in resultados:
        print(f"\n{nombre}:")
        _edp_print_convergencia(hist)


def menu_graficar_edp():
    if not _MPLOK:
        print("\n  matplotlib no disponible — instala con: pip install matplotlib"); return
    print("\n=== GRAFICAR EDP ===")
    print("\nResuelve y grafica una EDP de Laplace o Poisson.")
    print("\nTipo de ecuacion:")
    print("  1. Laplace   — Gauss-Seidel")
    print("  2. Laplace   — SOR")
    print("  3. Poisson   — Gauss-Seidel")
    print("  4. Poisson   — SOR")
    print("  0. Cancelar")
    try:
        op = _input("\nElige [0-4]: ").strip()
    except (KeyboardInterrupt, EOFError, VolverAtras):
        return
    if op == "0": return
    try:
        op = int(op)
    except Exception:
        print("Opcion invalida"); return
    if op not in (1, 2, 3, 4):
        print("Opcion invalida"); return
    datos = _edp_datos()
    if datos is None: return
    xa, xb, ya, yb, Nx, Ny, bc_bot, bc_top, bc_left, bc_right = datos
    es_poisson = op in (3, 4)
    usa_sor = op in (2, 4)
    if es_poisson:
        print("\nTermino fuente f(x,y):")
        while True:
            raw = _pedir_texto("  f(x,y) = ")
            try:
                evaluar_src = crear_evaluador_seguro(raw, ("x", "y"))
                def f_src(x, y, _ev=evaluar_src): return _ev(x=x, y=y)
                f_src(0.0, 0.0)
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        def f_src(x, y): return 0.0
    if usa_sor:
        rho = 0.5 * (math.cos(math.pi / Nx) + math.cos(math.pi / Ny))
        omega_opt = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))
        omega = _pedir_float(f"\nomega (Enter = {omega_opt:.4f}): ",
                             permitir_vacio=True, default=omega_opt)
    else:
        omega = 1.0
    tol, maxit = _pedir_tol_iter(1e-5, 10000)
    xs = np.linspace(xa, xb, Nx + 1)
    ys = np.linspace(ya, yb, Ny + 1)
    hx = (xb - xa) / Nx
    hy = (yb - ya) / Ny
    u = np.zeros((Nx + 1, Ny + 1))
    for i in range(Nx + 1):
        u[i, 0] = bc_bot(xs[i]); u[i, Ny] = bc_top(xs[i])
    for j in range(Ny + 1):
        u[0, j] = bc_left(ys[j]); u[Nx, j] = bc_right(ys[j])
    nombre = "Poisson" if es_poisson else "Laplace"
    metodo = f"SOR (omega={omega:.4f})" if usa_sor else "Gauss-Seidel"
    print(f"\nResolviendo {nombre} — {metodo}  (grid {Nx+1}x{Ny+1})...")
    it, delta = _edp_solver(u, xs, ys, hx, hy, Nx, Ny, f_src, omega, tol, maxit)
    if delta < tol:
        print(f"Convergencia en {it} iteraciones  (delta = {delta:.3e})")
    else:
        print(f"No convergio en {maxit} iteraciones  (delta = {delta:.3e})")
    _gr.edp(u, xs, ys, titulo=f"{nombre} — {metodo}")


def menu_edp():
    while True:
        print("\n=== ECUACIONES EN DERIVADAS PARCIALES (TEMA 6) ===\n")
        print("  Diferencias finitas 5 puntos — condiciones Dirichlet")
        print()
        print("  1. Laplace   nabla^2 u = 0        — Gauss-Seidel (Liebmann)")
        print("  2. Laplace   nabla^2 u = 0        — SOR (sobre-relajacion)")
        print("  3. Poisson   nabla^2 u = f(x,y)   — Gauss-Seidel")
        print("  4. Poisson   nabla^2 u = f(x,y)   — SOR")
        print("  5. Graficar")
        print("  6. Comparar Gauss-Seidel vs SOR")
        print("  0. Volver al menu principal")
        try:
            op = _input("\nElige [0-6]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op == "0": break
        try:
            op = int(op)
        except Exception:
            print("Opcion invalida"); continue
        if op not in (1, 2, 3, 4, 5, 6):
            print("Opcion invalida"); continue
        if op == 5:
            try: menu_graficar_edp()
            except VolverAtras: continue
        elif op == 6:
            try: _edp_comparar_gs_sor()
            except VolverAtras: continue
        else:
            try: _edp_run(es_poisson=(op in (3, 4)), usa_sor=(op in (2, 4)))
            except VolverAtras: continue




__all__ = [name for name in globals() if not name.startswith("__")]
