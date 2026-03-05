import math
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _normalizar_expr(s: str) -> str:
    """Normaliza sintaxis común: ^ -> **, y deja ln(...) tal cual (se trata luego)."""
    s = s.strip()
    s = s.replace("^", "**")
    return s


def _derivada_simbolica(expr_str):
    """
    Calcula f'(x) con sympy. Devuelve (df_func, df_str).
    Requiere: pip install sympy
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
            "No puedo calcular la derivada porque falta 'sympy'. "
            "Instálalo con: pip install sympy"
        ) from e

    x = symbols("x")
    s = _normalizar_expr(expr_str)

    # Para sympy:
    s = s.replace("ln(", "log(")
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bpi\b", "pi", s)
    s = re.sub(r"\be\b", "E", s)          # si el usuario escribe e
    s = re.sub(r"\bmath\.", "", s)        # permitir sin(x) sin prefijo

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)
    df_expr = diff(expr, x)

    df_func = lambdify(x, df_expr, modules=["math", "numpy"])
    return df_func, str(df_expr)


def _crear_funcion_segura(f_str):
    """
    Crea f(x) para eval:
    - acepta ^, ln(x)
    - acepta sin(x), cos(x), exp(x), log(x), sqrt(x) sin prefijo
    - acepta pi y e
    """
    expr = _normalizar_expr(f_str)

    # Para eval (math):
    expr = expr.replace("ln(", "math.log(")

    allowed_globals = {
        "__builtins__": {},
        "math": math,
        "np": np,
        # funciones comunes sin prefijo
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        # constantes
        "pi": math.pi,
        "e": math.e,
    }

    def f(x):
        return eval(expr, allowed_globals, {"x": x})

    return f


def metodo_newton_raphson(f, df, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Newton-Raphson con ERROR RELATIVO típico (respecto a x_n):

        error_rel = |x_{n+1} - x_n| / |x_n|

    Si |x_n| ~ 0, cae a error absoluto.
    Criterio de parada: SOLO error_rel < tolerancia (para que sea comparable).
    """
    print("\n{:<10} {:<22} {:<22} {:<22} {:<22}".format(
        "Iteración", "x_n", "f(x_n)", "f'(x_n)", "Error rel."
    ))
    print("-" * 110)

    x_actual = x0
    historial_x = [x0]
    iteracion = 0
    error_rel = float("inf")

    while iteracion < max_iteraciones:
        try:
            fx = f(x_actual)
            dfx = df(x_actual)
        except Exception as e:
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None, historial_x

        if not (np.isfinite(fx) and np.isfinite(dfx)):
            print(f"\nError: f(x) o f'(x) no es finito en x = {x_actual}.")
            return None, historial_x

        if abs(dfx) < 1e-15:
            print(f"\nError: La derivada es cero (o muy cercana a cero) en x = {x_actual}.")
            return None, historial_x

        x_siguiente = x_actual - fx / dfx

        salto = abs(x_siguiente - x_actual)
        denom = abs(x_actual)  # relativo respecto a x_n
        error_rel = (salto / denom) if denom > 1e-15 else salto

        print("{:<10d} {:<22.10f} {:<22.10e} {:<22.10e} {:<22.10e}".format(
            iteracion, x_actual, fx, dfx, error_rel
        ))

        historial_x.append(x_siguiente)

        if error_rel < tolerancia:
            fx_siguiente = f(x_siguiente)
            print(f"\n✓ Raíz encontrada: x = {x_siguiente:.10f}")
            print(f"✓ f({x_siguiente:.10f}) = {fx_siguiente:.10e}")
            print(f"✓ Error relativo: {error_rel:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente, historial_x

        x_actual = x_siguiente
        iteracion += 1

    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    fx_final = f(x_actual)
    print(f"Raíz aproximada: x = {x_actual:.10f}")
    print(f"f({x_actual:.10f}) = {fx_final:.10e}")
    print(f"Error relativo: {error_rel:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return x_actual, historial_x


def ingresar_funcion():
    print("\n=== MÉTODO DE NEWTON-RAPHSON (error relativo) ===\n")
    print("  x_{n+1} = x_n - f(x_n) / f'(x_n)\n")
    print("Puedes usar: sin(x), cos(x), exp(x), log(x), sqrt(x), pi, e, ^, ln(x)\n")

    f_str = input("f(x) = ").strip()

    try:
        f = _crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    try:
        df, df_str = _derivada_simbolica(f_str)
        df(1)
        print(f"f'(x) calculada automáticamente: {df_str}\n")
    except Exception as e:
        print(f"Error al calcular la derivada automáticamente: {e}")
        return None, None, None, None

    return f, f_str, df, df_str


def graficar_resultado(f, f_str, df, raiz, x0, historial_x):
    todos_x = [x for x in (historial_x + [raiz]) if x is not None and np.isfinite(x)]
    if not todos_x:
        print("No hay puntos válidos para graficar.")
        return

    rango = max(abs(max(todos_x) - raiz), abs(min(todos_x) - raiz), 1.0)
    margen = rango * 1.4
    x_min = raiz - margen
    x_max = raiz + margen

    x_vals = np.linspace(x_min, x_max, 600)
    f_vals = []
    for x in x_vals:
        try:
            y = f(x)
            f_vals.append(y if np.isfinite(y) else np.nan)
        except Exception:
            f_vals.append(np.nan)
    f_vals = np.array(f_vals, dtype=float)

    try:
        f_raiz = f(raiz)
    except Exception:
        f_raiz = np.nan

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, f_vals, "b-", linewidth=2, label=f"f(x) = {f_str}")
    plt.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.3)
    plt.axvline(x=raiz, color="g", linestyle="--", linewidth=1, alpha=0.3)

    # Tangentes (primeras 8 iteraciones)
    iteraciones_a_mostrar = historial_x[:-1][:8]
    colores_tangente = plt.cm.Oranges(np.linspace(0.4, 0.9, max(len(iteraciones_a_mostrar), 1)))

    for i, x_n in enumerate(iteraciones_a_mostrar):
        try:
            fx_n = f(x_n)
            dfx_n = df(x_n)
            if not (np.isfinite(fx_n) and np.isfinite(dfx_n)) or abs(dfx_n) < 1e-15:
                continue
            tang_x = np.array([x_n - margen * 0.4, x_n + margen * 0.4])
            tang_y = fx_n + dfx_n * (tang_x - x_n)
            label = "Tangentes" if i == 0 else None
            plt.plot(tang_x, tang_y, "-", color=colores_tangente[i], linewidth=1.2, alpha=0.75, label=label)
            plt.plot(x_n, fx_n, "o", color=colores_tangente[i], markersize=7, zorder=4)
        except Exception:
            continue

    # Punto inicial
    try:
        y0 = f(x0)
        if np.isfinite(y0):
            plt.plot(x0, y0, "gs", markersize=10, label=f"Inicio: x₀ = {x0}", zorder=5)
    except Exception:
        pass

    # Punto raíz
    if np.isfinite(f_raiz):
        plt.plot(raiz, f_raiz, "ro", markersize=12, label="Raíz encontrada", zorder=6)

    # Anotación con error relativo final
    num_iteraciones = len(historial_x) - 1
    if len(historial_x) >= 2:
        salto_final = abs(historial_x[-1] - historial_x[-2])
        denom_final = abs(historial_x[-2])
        err_rel_final = (salto_final / denom_final) if denom_final > 1e-15 else salto_final
    else:
        err_rel_final = 0.0

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red", linewidth=2)
    anotacion = f"x = {raiz:.8f}\nf(x) = {f_raiz:.2e}\nErr rel.: {err_rel_final:.2e}\nIteraciones: {num_iteraciones}"

    f_visible = f_vals[~np.isnan(f_vals)]
    y_range = max(float(f_visible.max() - f_visible.min()), 1.0) if len(f_visible) > 0 else 1.0
    offset_y = y_range * 0.15

    if np.isfinite(f_raiz):
        plt.annotate(
            anotacion,
            xy=(raiz, f_raiz),
            xytext=(raiz, f_raiz + offset_y),
            bbox=bbox_props,
            fontsize=11,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

    plt.grid(True, alpha=0.3)
    plt.xlabel("x", fontsize=12, fontweight="bold")
    plt.ylabel("f(x)", fontsize=12, fontweight="bold")
    plt.title(f"Método de Newton-Raphson (error relativo)\nf(x) = {f_str}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    nombre_archivo = "grafica_newton_raphson.png"
    plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada como: {nombre_archivo}")

    try:
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
    except Exception:
        print("(No se pudo abrir la ventana de la gráfica, pero se guardó el archivo)")
    finally:
        plt.close()


def main():
    f, f_str, df, df_str = ingresar_funcion()
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
    raiz, historial_x = metodo_newton_raphson(f, df, x0, tolerancia, max_iteraciones)

    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(f, f_str, df, raiz, x0, historial_x)


if __name__ == "__main__":
    main()