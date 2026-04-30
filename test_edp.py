import sys
import math
import numpy as np

sys.path.insert(0, '/Users/luichi/Documents/Programas-aproximaci-n-numerica')
import main

def make_solver(xa, xb, ya, yb, N, bc_bot, bc_top, bc_left, bc_right,
                f_src=None, omega=1.0, tol=1e-10, maxit=50000):
    if f_src is None:
        f_src = lambda x, y: 0.0
    xs = np.linspace(xa, xb, N + 1)
    ys = np.linspace(ya, yb, N + 1)
    hx = (xb - xa) / N
    hy = (yb - ya) / N
    u = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        u[i, 0] = bc_bot(xs[i])
        u[i, N] = bc_top(xs[i])
    for j in range(N + 1):
        u[0, j] = bc_left(ys[j])
        u[N, j] = bc_right(ys[j])
    it, delta = main._edp_solver(u, xs, ys, hx, hy, N, f_src, omega, tol, maxit)
    return u, xs, ys, it, delta


results = []

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")


# ─────────────────────────────────────────────
# TEST 1: Laplace [0,1]² — solución u=x*y
# ─────────────────────────────────────────────
N = 10
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: x,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: y,
    tol=1e-10
)
max_err = 0.0
for i in range(N + 1):
    for j in range(N + 1):
        max_err = max(max_err, abs(u[i, j] - xs[i] * ys[j]))
passed = max_err < 1e-6
report("Test 1 — Laplace u=x*y (solución exacta)", passed,
       f"error máx = {max_err:.3e}, iters = {it}")

# ─────────────────────────────────────────────
# TEST 2: Laplace — BCs constantes, simetría
# ─────────────────────────────────────────────
N = 4
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: 100.0,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: 0.0,
    tol=1e-10
)
converged = delta < 1e-10
sym_ok = all(abs(u[1, j] - u[N-1, j]) < 1e-8 for j in range(N + 1))
passed = converged and sym_ok
report("Test 2 — Laplace BCs constantes, simetría", passed,
       f"convergió={converged}, simetría={sym_ok}, iters={it}")

# ─────────────────────────────────────────────
# TEST 3: Laplace — u=x²-y² (armónica: ∇²u = 2 - 2 = 0)
# ─────────────────────────────────────────────
N = 6
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: x**2,
    bc_top   = lambda x: x**2 - 1,
    bc_left  = lambda y: -y**2,
    bc_right = lambda y: 1 - y**2,
    tol=1e-10
)
max_err = 0.0
for i in range(N + 1):
    for j in range(N + 1):
        max_err = max(max_err, abs(u[i, j] - (xs[i]**2 - ys[j]**2)))
passed = max_err < 1e-4
report("Test 3 — Laplace u=x2-y2 (armonico, solucion exacta)", passed,
       f"error max = {max_err:.3e}, iters = {it}")

# ─────────────────────────────────────────────
# TEST 4: Poisson f=-2, u=0 borde
# ─────────────────────────────────────────────
N = 4
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: 0.0,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: 0.0,
    f_src    = lambda x, y: -2.0,
    tol=1e-10
)
interior = [u[i, j] for i in range(1, N) for j in range(1, N)]
pos = all(v > 0 for v in interior)
u_min = min(interior)
u_max = max(interior)
passed = pos and delta < 1e-10
report("Test 4 — Poisson f=-2, interior positivo", passed,
       f"min={u_min:.4f}, max={u_max:.4f}, iters={it}")

# ─────────────────────────────────────────────
# TEST 5: Poisson — solución exacta u=x(1-x)/2 + y(1-y)/2
# ∇²u = d²/dx²[x(1-x)/2] + d²/dy²[y(1-y)/2] = -1 + -1 = -2 ✓
# BCs: u=0 en borde (ya que en x=0,1 e y=0,1 la función vale 0)
# ─────────────────────────────────────────────
N = 8
def exact5(x, y): return x*(1-x)/2 + y*(1-y)/2

u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: exact5(x, 0),
    bc_top   = lambda x: exact5(x, 1),
    bc_left  = lambda y: exact5(0, y),
    bc_right = lambda y: exact5(1, y),
    f_src    = lambda x, y: -2.0,
    tol=1e-10
)
max_err = 0.0
for i in range(1, N):
    for j in range(1, N):
        max_err = max(max_err, abs(u[i, j] - exact5(xs[i], ys[j])))
passed = max_err < 1e-4
report("Test 5 — Poisson u=x(1-x)/2+y(1-y)/2 (solucion exacta, nabla2u=-2)", passed,
       f"error max = {max_err:.3e}, iters = {it}")

# ─────────────────────────────────────────────
# TEST 6: SOR vs Gauss-Seidel — mismas BCs test 1
# ─────────────────────────────────────────────
N = 10
tol6 = 1e-8

_, _, _, it_gs, _ = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: x,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: y,
    omega=1.0, tol=tol6
)

rho = math.cos(math.pi / N)
omega_opt = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))
_, _, _, it_sor, _ = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: x,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: y,
    omega=omega_opt, tol=tol6
)

passed = it_sor < it_gs
report("Test 6 — SOR converge en menos iteraciones que GS", passed,
       f"GS={it_gs} iters, SOR={it_sor} iters (omega={omega_opt:.4f})")

# ─────────────────────────────────────────────
# TEST 7: Dominio no cuadrado [0,2]×[0,1]
# ─────────────────────────────────────────────
N = 6
xa, xb, ya, yb = 0, 2, 0, 1
xs_ref = np.linspace(xa, xb, N + 1)
ys_ref = np.linspace(ya, yb, N + 1)
hx_ref = (xb - xa) / N
hy_ref = (yb - ya) / N

u, xs, ys, it, delta = make_solver(
    xa, xb, ya, yb, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: 1.0,
    bc_left  = lambda y: y,
    bc_right = lambda y: y,
    tol=1e-8
)
hx_ok = abs((xs[1] - xs[0]) - hx_ref) < 1e-14
hy_ok = abs((ys[1] - ys[0]) - hy_ref) < 1e-14
converged = delta < 1e-8
passed = hx_ok and hy_ok and converged
report("Test 7 — Dominio no cuadrado [0,2]×[0,1]", passed,
       f"hx={xs[1]-xs[0]:.6f} (esp {hx_ref:.6f}), hy={ys[1]-ys[0]:.6f} (esp {hy_ref:.6f}), iters={it}")

# ─────────────────────────────────────────────
# TEST 8: Laplace en [0,π]² — BC sin(x)
# ─────────────────────────────────────────────
N = 6
pi = math.pi
u, xs, ys, it, delta = make_solver(
    0, pi, 0, pi, N,
    bc_bot   = lambda x: math.sin(x),
    bc_top   = lambda x: -math.sin(x),
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: 0.0,
    tol=1e-8
)
max_err_bc = max(abs(u[i, 0] - math.sin(xs[i])) for i in range(N + 1))
passed = max_err_bc < 1e-12 and delta < 1e-8
report("Test 8 — BCs sin(x), j=0 = sin(xs[i])", passed,
       f"error en BC inferior = {max_err_bc:.3e}, iters={it}")

# ─────────────────────────────────────────────
# TEST 9: N grande (N=20), SOR
# ─────────────────────────────────────────────
import time
N = 20
rho9 = math.cos(math.pi / N)
omega9 = 2.0 / (1.0 + math.sqrt(1.0 - rho9 * rho9))

t0 = time.time()
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: 100.0,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: 0.0,
    omega=omega9, tol=1e-6, maxit=50000
)
elapsed = time.time() - t0
passed = delta < 1e-6 and elapsed < 5.0
report("Test 9 — N=20 SOR, tiempo < 5s", passed,
       f"iters={it}, delta={delta:.3e}, tiempo={elapsed:.2f}s")

# ─────────────────────────────────────────────
# TEST 10: Poisson f=sin(πx)sin(πy) — centro negativo
# ─────────────────────────────────────────────
N = 8
u, xs, ys, it, delta = make_solver(
    0, 1, 0, 1, N,
    bc_bot   = lambda x: 0.0,
    bc_top   = lambda x: 0.0,
    bc_left  = lambda y: 0.0,
    bc_right = lambda y: 0.0,
    f_src    = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y),
    tol=1e-10
)
mid = N // 2
u_center = u[mid, mid]
passed = u_center < 0
report("Test 10 — Poisson f=sin(πx)sin(πy), centro < 0", passed,
       f"u(0.5,0.5) = {u_center:.6f}, iters={it}")

# ─────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("RESUMEN DE TESTS")
print("═" * 60)
passed_count = sum(1 for _, s, _ in results if s == "PASS")
for name, status, detail in results:
    print(f"  [{status}] {name}")
print(f"\n  Total: {passed_count}/{len(results)} PASS")
print("═" * 60)
