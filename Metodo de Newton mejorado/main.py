import math
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Utilidades de parsing / eval
# -----------------------------
def _normalizar_expr(s: str) -> str:
    """Normaliza sintaxis común: ^ -> **."""
    s = s.strip()
    s = s.replace("^", "**")
    return s


def _derivadas_simbolicas(expr_str):
    """
    Calcula f'(x) y f''(x) con sympy.
    Devuelve (df_func, df_str, d2f_func, d2f_str).
    """
    try:
        from sympy import symbols, diff, lambdify
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except Exception as e:
        raise RuntimeError(
            "Falta sympy. Instálalo con: pip install sympy"
        ) from e

    x = symbols("x")

    s = _normalizar_expr(expr_str)
    s = s.replace("ln(", "log(")
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bpi\b", "pi", s)
    s = re.sub(r"\be\b", "E", s)
    s = re.sub(r"\bmath\.", "", s)

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    df_expr = diff(expr, x)
    d2f_expr = diff(df_expr, x)

    df_func = lambdify(x, df_expr, modules=["math", "numpy"])
    d2f_func = lambdify(x, d2f_expr, modules=["math", "numpy"])

    return df_func, str(df_expr), d2f_func, str(d2f_expr)


def _crear_funcion_segura(f_str):
    """
    Crea f(x) para eval:
    - acepta ^, ln(x)
    - acepta sin(x), cos(x), exp(x), log(x), sqrt(x) sin prefijo
    - acepta pi y e
    """
    expr = _normalizar_expr(f_str)
    expr = expr.replace("ln(", "math.log(")

    allowed_globals = {
        "__builtins__": {},
        "math": math,
        "np": np,
        "abs": abs,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    def f(x):
        return eval(expr, allowed_globals, {"x": x})

    return f


# -----------------------------
# Newton mejorado (con f' y f'')
# -----------------------------
def metodo_newton_mejorado(f, df, d2f, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Newton mejorado clásico (usa f', f''):

      x_{n+1} = x_n - (f * f') / ( (f')^2 - f * f'' )

    Error relativo (como antes):
      error_rel = |x_{n+1}-x_n| / |x_n|  (si |x_n|~0 -> absoluto)

    Parada: SOLO por error_rel < tolerancia (para comparar con compañeros).
    """

    print("\n{:<10} {:<22} {:<22} {:<22} {:<22}".format(
        "Iteración", "x_n", "f(x_n)", "Paso", "Error rel."
    ))
    print("-" * 105)

    x_actual = x0
    historial_x = [x0]
    error_rel = float("inf")

    for iteracion in range(max_iteraciones):
        try:
            fx = f(x_actual)
            dfx = df(x_actual)
            d2fx = d2f(x_actual)
        except Exception as e:
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None, historial_x

        if not (np.isfinite(fx) and np.isfinite(dfx) and np.isfinite(d2fx)):
            print(f"\nError: f(x), f'(x) o f''(x) no es finito en x = {x_actual}.")
            return None, historial_x

        denom = (dfx * dfx) - (fx * d2fx)
        if abs(denom) < 1e-15:
            print(f"\nError: Denominador ~ 0 en x = {x_actual}. No se puede continuar.")
            return None, historial_x

        paso = (fx * dfx) / denom
        x_siguiente = x_actual - paso

        # Error relativo respecto a x_n (x_actual)
        salto = abs(x_siguiente - x_actual)
        denom_err = abs(x_actual)
        error_rel = (salto / denom_err) if denom_err > 1e-15 else salto

        print("{:<10d} {:<22.10f} {:<22.10e} {:<22.10e} {:<22.10e}".format(
            iteracion, x_actual, fx, paso, error_rel
        ))

        historial_x.append(x_siguiente)

        if error_rel < tolerancia:
            fx_sig = f(x_siguiente)
            print(f"\n✓ Raíz encontrada: x = {x_siguiente:.10f}")
            print(f"✓ f({x_siguiente:.10f}) = {fx_sig:.10e}")
            print(f"✓ Error relativo: {error_rel:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente, historial_x

        x_actual = x_siguiente

    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    fx_final = f(x_actual)
    print(f"Raíz aproximada: x = {x_actual:.10f}")
    print(f"f({x_actual:.10f}) = {fx_final:.10e}")
    print(f"Error relativo: {error_rel:.10e}")
    return x_actual, historial_x


# -----------------------------
# UI (entrada / gráfica)
# -----------------------------
def ingresar_funcion():
    print("\n=== MÉTODO DE NEWTON MEJORADO (con f' y f'') ===\n")
    print("Fórmula:")
    print("  x_{n+1} = x_n - (f(x_n) f'(x_n)) / ((f'(x_n))^2 - f(x_n) f''(x_n))\n")
    print("Puedes escribir: sin(x), cos(x), exp(x), log(x), sqrt(x), pi, e, ^, ln(x)\n")

    f_str = input("f(x) = ").strip()

    try:
        f = _crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None, None, None

    try:
        df, df_str, d2f, d2f_str = _derivadas_simbolicas(f_str)
        df(1); d2f(1)
        print(f"f'(x):  {df_str}")
        print(f"f''(x): {d2f_str}\n")
    except Exception as e:
        print(f"Error al calcular derivadas automáticamente: {e}")
        return None, None, None, None, None, None

    return f, f_str, df, df_str, d2f, d2f_str


def graficar_resultado(f, f_str, raiz, x0, historial_x):
    todos_x = [x for x in (historial_x + [raiz]) if x is not None and np.isfinite(x)]
    if not todos_x:
        print("No hay puntos válidos para graficar.")
        return

    rango = max(abs(max(todos_x) - raiz), abs(min(todos_x) - raiz), 1.0)
    margen = rango * 1.4
    x_min = raiz - margen
    x_max = raiz + margen

    x_vals = np.linspace(x_min, x_max, 600)
    y_vals = []
    for x in x_vals:
        try:
            y = f(x)
            y_vals.append(y if np.isfinite(y) else np.nan)
        except Exception:
            y_vals.append(np.nan)
    y_vals = np.array(y_vals, dtype=float)

    f_raiz = f(raiz)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, "b-", linewidth=2, label=f"f(x) = {f_str}")
    plt.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.3)
    plt.axvline(raiz, color="g", linestyle="--", linewidth=1, alpha=0.3)

    # puntos de iteración (primeras 12)
    pts = historial_x[:12]
    ys = []
    for x in pts:
        try:
            ys.append(f(x))
        except Exception:
            ys.append(np.nan)
    plt.plot(pts, ys, "ko", markersize=4, alpha=0.6, label="Iteraciones")

    # inicio y raíz
    try:
        plt.plot(x0, f(x0), "gs", markersize=10, label=f"Inicio x₀={x0}", zorder=5)
    except Exception:
        pass
    plt.plot(raiz, f_raiz, "ro", markersize=10, label="Raíz", zorder=6)

    plt.grid(True, alpha=0.3)
    plt.xlabel("x", fontsize=12, fontweight="bold")
    plt.ylabel("f(x)", fontsize=12, fontweight="bold")
    plt.title("Newton Mejorado (con f' y f'')\n" + f"f(x) = {f_str}", fontsize=14, fontweight="bold")
    plt.legend(loc="best")
    plt.tight_layout()

    nombre_archivo = "grafica_newton_mejorado.png"
    plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada como: {nombre_archivo}")

    try:
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
    except Exception:
        print("(No se pudo abrir la ventana, pero se guardó el archivo)")
    finally:
        plt.close()


def main():
    f, f_str, df, df_str, d2f, d2f_str = ingresar_funcion()
    if f is None:
        return

    print("\nIngresa el valor inicial x₀:")
    try:
        x0 = float(input("x0 = "))
    except ValueError:
        print("Error: Debes ingresar un número válido")
        return

    try:
        tolerancia_str = input("\nTolerancia (Enter para 1e-3): ").strip()
        tolerancia = float(tolerancia_str) if tolerancia_str else 1e-3
        if tolerancia <= 0:
            raise ValueError
    except Exception:
        print("Error en la tolerancia, usando 1e-3")
        tolerancia = 1e-3

    try:
        max_iter_str = input("Máx. iteraciones (Enter para 100): ").strip()
        max_iteraciones = int(max_iter_str) if max_iter_str else 100
        if max_iteraciones <= 0:
            raise ValueError
    except Exception:
        print("Error en iteraciones, usando 100")
        max_iteraciones = 100

    print(f"\nResolviendo: f(x) = {f_str}, con x₀ = {x0}")
    raiz, historial_x = metodo_newton_mejorado(f, df, d2f, x0, tolerancia, max_iteraciones)

    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(f, f_str, raiz, x0, historial_x)


if __name__ == "__main__":
    main()