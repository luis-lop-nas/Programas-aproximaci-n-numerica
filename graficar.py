"""graficar.py — Visualizacion de resultados de metodos numericos."""

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _OK = True
except ImportError:
    _OK = False


def disponible():
    return _OK


def _check():
    if not _OK:
        print("  matplotlib no disponible. Instala con:  pip install matplotlib")
    return _OK


def _titulo(fig, t):
    try:
        fig.canvas.manager.set_window_title(t)
    except Exception:
        pass


# ── Raices ─────────────────────────────────────────────────────────────────────

def raiz_fx(f, a, b, raiz, f_str, metodo=""):
    """f(x) en [a,b] con la raiz marcada."""
    if not _check():
        return
    xs = np.linspace(a, b, 600)
    ys = np.array([_eval(f, x) for x in xs])
    fig, ax = plt.subplots()
    _titulo(fig, f"Raiz — {metodo}")
    ax.plot(xs, ys, 'b-', linewidth=1.5, label=f'f(x) = {f_str}')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    if raiz is not None:
        ax.axvline(raiz, color='r', linewidth=1, linestyle=':', alpha=0.7)
        ax.plot(raiz, 0, 'ro', markersize=8, label=f'Raiz  x = {raiz:.8g}')
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    ax.set_title(f'{metodo}   f(x) = {f_str}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def raiz_convergencia(hist_en_dict):
    """Convergencia de uno o varios metodos. hist_en_dict = {nombre: [en0, en1, ...]}."""
    if not _check():
        return
    fig, ax = plt.subplots()
    _titulo(fig, "Convergencia — error relativo")
    for nombre, ens in hist_en_dict.items():
        vals = [e for e in ens if e is not None and e > 0 and np.isfinite(e)]
        if vals:
            ax.semilogy(range(len(vals)), vals, marker='o', markersize=4, label=nombre)
    ax.set_xlabel('Iteracion'); ax.set_ylabel('Error relativo EN')
    ax.set_title('Convergencia'); ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.show()


def raiz_comparacion(f, a, b, f_str, datos_list):
    """
    Compara varios metodos en la misma funcion.
    datos_list: [{'metodo': str, 'raiz': float, 'hist_en': list}, ...]
    """
    if not _check():
        return
    xs = np.linspace(a, b, 600)
    ys = np.array([_eval(f, x) for x in xs])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _titulo(fig, "Comparacion de metodos — Raices")
    colores = ['r', 'g', 'm', 'orange', 'c', 'brown']

    axes[0].plot(xs, ys, 'b-', linewidth=1.5, label=f'f(x) = {f_str}')
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    for i, d in enumerate(datos_list):
        c = colores[i % len(colores)]
        if d.get('raiz') is not None:
            axes[0].axvline(d['raiz'], color=c, linewidth=1.5, linestyle=':', label=d['metodo'])
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].set_title('f(x) con raices'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for i, d in enumerate(datos_list):
        c = colores[i % len(colores)]
        ens = [e for e in d.get('hist_en', []) if e is not None and e > 0 and np.isfinite(e)]
        if ens:
            axes[1].semilogy(range(len(ens)), ens, marker='o', markersize=3, color=c, label=d['metodo'])
    axes[1].set_xlabel('Iteracion'); axes[1].set_ylabel('EN')
    axes[1].set_title('Convergencia'); axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

    plt.suptitle(f'f(x) = {f_str}  en  [{a:.4g}, {b:.4g}]')
    plt.tight_layout(); plt.show()


# ── Interpolacion / Regresion ───────────────────────────────────────────────────

def interpolacion(x_datos, y_datos, curvas_list, titulo="Ajuste"):
    """
    Grafica puntos de datos y una o varias curvas ajustadas.
    curvas_list: [{'xs': arr, 'ys': arr, 'label': str}, ...]
    """
    if not _check():
        return
    fig, ax = plt.subplots()
    _titulo(fig, titulo)
    ax.scatter(x_datos, y_datos, color='k', zorder=5, s=45, marker='x', label='Datos')
    for c in curvas_list:
        ax.plot(c['xs'], c['ys'], linewidth=1.5, label=c['label'])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(titulo); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ── Derivacion ──────────────────────────────────────────────────────────────────

def derivacion(x_datos, y_datos, x_deriv, y_aprox, y_exact, titulo, etiqueta):
    """Grafica datos originales y la derivada aproximada vs exacta (Lagrange)."""
    if not _check():
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _titulo(fig, titulo)
    axes[0].scatter(x_datos, y_datos, color='k', s=30, zorder=5, label='(xi, yi)')
    axes[0].plot(x_datos, y_datos, 'b--', linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_title('Datos')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_deriv, y_aprox, 'bo-', markersize=4, linewidth=1.2, label=f'{etiqueta} (aprox.)')
    if y_exact is not None:
        axes[1].plot(x_deriv, y_exact, 'r--', linewidth=1.5, label=f'{etiqueta} (Lagrange)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel(etiqueta)
    axes[1].set_title(titulo); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ── Integracion ─────────────────────────────────────────────────────────────────

def integracion(f, a, b, n, metodo, resultado):
    """f(x) con area sombreada y nodos del metodo marcados."""
    if not _check():
        return
    xs_fine = np.linspace(a, b, 600)
    ys_fine = np.array([_eval(f, x) for x in xs_fine])
    n_plot = max(n if n else 2, 2)
    xs_n = np.linspace(a, b, n_plot + 1)
    ys_n = np.array([_eval(f, x) for x in xs_n])

    fig, ax = plt.subplots()
    _titulo(fig, f"Integracion — {metodo}")
    ax.plot(xs_fine, ys_fine, 'b-', linewidth=1.5, label='f(x)')
    ax.fill_between(xs_fine, 0, ys_fine, alpha=0.15, color='steelblue',
                    label=f'I ≈ {resultado:.6g}')
    ax.scatter(xs_n, ys_n, color='r', zorder=5, s=25, label=f'Nodos (n={n_plot})')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    ax.set_title(f'[{a:.4g}, {b:.4g}]   I ≈ {resultado:.6g}   ({metodo})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def integracion_comparacion(f, a, b, resultados, f_str="f"):
    """
    Compara varios metodos de integracion.
    resultados: [{'metodo': str, 'resultado': float, 'n': int}, ...]
    """
    if not _check():
        return
    xs_fine = np.linspace(a, b, 600)
    ys_fine = np.array([_eval(f, x) for x in xs_fine])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _titulo(fig, "Comparacion de integracion")

    axes[0].plot(xs_fine, ys_fine, 'b-', linewidth=1.5, label=f'{f_str}')
    axes[0].fill_between(xs_fine, 0, ys_fine, alpha=0.12, color='steelblue')
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].set_title(f'Funcion en [{a:.4g}, {b:.4g}]')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    etiquetas = [d['metodo'] for d in resultados]
    valores = [d['resultado'] for d in resultados]
    axes[1].bar(range(len(valores)), valores, color='steelblue', alpha=0.75)
    axes[1].set_xticks(range(len(valores)))
    axes[1].set_xticklabels(etiquetas, rotation=25, ha='right')
    axes[1].set_ylabel('Integral aproximada')
    axes[1].set_title('Resultados por metodo')
    axes[1].grid(True, axis='y', alpha=0.3)
    for i, val in enumerate(valores):
        axes[1].text(i, val, f'{val:.6g}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Integracion de {f_str}')
    plt.tight_layout(); plt.show()


# ── EDOs ────────────────────────────────────────────────────────────────────────

def edo_escalar(datos_list, eje_x='x', eje_y='y(x)', titulo="EDO escalar"):
    """
    Uno o varios metodos para la misma EDO escalar.
    datos_list: [{'xs': arr, 'ys': arr, 'label': str}, ...]
    """
    if not _check():
        return
    fig, ax = plt.subplots()
    _titulo(fig, titulo)
    for d in datos_list:
        ax.plot(d['xs'], d['ys'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
    ax.set_xlabel(eje_x); ax.set_ylabel(eje_y)
    ax.set_title(titulo); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def edo_sistema(datos_list, eje, titulo="Sistema 2 EDOs"):
    """
    Uno o varios metodos para un sistema 2 EDOs.
    datos_list: [{'t': arr, 'x': arr, 'y': arr, 'label': str}, ...]
    eje: 'xt' | 'yt' | 'xy'
    """
    if not _check():
        return
    if eje == 'ambas':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _titulo(fig, titulo)
        for d in datos_list:
            axes[0].plot(d['t'], d['x'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
            axes[1].plot(d['t'], d['y'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        axes[0].set_xlabel('t'); axes[0].set_ylabel('x(t)')
        axes[1].set_xlabel('t'); axes[1].set_ylabel('y(t)')
        for a2 in axes: a2.legend(); a2.grid(True, alpha=0.3)
        plt.suptitle(titulo); plt.tight_layout(); plt.show()
        return
    fig, ax = plt.subplots()
    _titulo(fig, titulo)
    for d in datos_list:
        if eje == 'xt':
            ax.plot(d['t'], d['x'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        elif eje == 'yt':
            ax.plot(d['t'], d['y'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        else:
            ax.plot(d['x'], d['y'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
    if eje == 'xt':   ax.set_xlabel('t'); ax.set_ylabel('x(t)')
    elif eje == 'yt': ax.set_xlabel('t'); ax.set_ylabel('y(t)')
    else:             ax.set_xlabel('x(t)'); ax.set_ylabel('y(t)')
    ax.set_title(titulo); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def edo_2orden(datos_list, eje, titulo="EDO 2° orden"):
    """
    Uno o varios metodos para una EDO de 2° orden.
    datos_list: [{'t': arr, 'x': arr, 'v': arr, 'label': str}, ...]
    eje: 'xt' | 'vt' | 'xv' | 'ambas'
    """
    if not _check():
        return
    if eje == 'ambas':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _titulo(fig, titulo)
        for d in datos_list:
            axes[0].plot(d['t'], d['x'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
            axes[1].plot(d['t'], d['v'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        axes[0].set_xlabel('t'); axes[0].set_ylabel("x(t)")
        axes[1].set_xlabel('t'); axes[1].set_ylabel("x'(t)")
        for a2 in axes: a2.legend(); a2.grid(True, alpha=0.3)
        plt.suptitle(titulo); plt.tight_layout(); plt.show()
        return
    fig, ax = plt.subplots()
    _titulo(fig, titulo)
    for d in datos_list:
        if eje == 'xt':
            ax.plot(d['t'], d['x'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        elif eje == 'vt':
            ax.plot(d['t'], d['v'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
        else:
            ax.plot(d['x'], d['v'], marker='.', markersize=2, linewidth=1.2, label=d['label'])
    if eje == 'xt':   ax.set_xlabel('t'); ax.set_ylabel("x(t)")
    elif eje == 'vt': ax.set_xlabel('t'); ax.set_ylabel("x'(t)")
    else:             ax.set_xlabel("x(t)"); ax.set_ylabel("x'(t)")
    ax.set_title(titulo); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ── EDP ─────────────────────────────────────────────────────────────────────────

def edp(u, xs, ys, titulo="EDP — u(x,y)"):
    """Mapa de calor y curvas de nivel de la solucion."""
    if not _check():
        return
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _titulo(fig, titulo)
    im = axes[0].pcolormesh(X, Y, u, shading='auto', cmap='hot')
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_title('Mapa de calor')
    c = axes[1].contourf(X, Y, u, levels=20, cmap='hot')
    plt.colorbar(c, ax=axes[1])
    axes[1].contour(X, Y, u, levels=20, colors='k', linewidths=0.3, alpha=0.5)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_title('Curvas de nivel')
    plt.suptitle(titulo); plt.tight_layout(); plt.show()


# ── Utilidad interna ─────────────────────────────────────────────────────────────

def _eval(f, x):
    try:
        v = float(f(x))
        return v if np.isfinite(v) else float('nan')
    except Exception:
        return float('nan')
