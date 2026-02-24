import math
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _derivada_simbolica(expr_str):
    """
    Calcula la derivada simbólica exacta de expr_str respecto a x usando sympy.
    Devuelve (función_derivada, string_derivada).
    """
    from sympy import symbols, diff, lambdify
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    x = symbols("x")

    # Normalizar sintaxis común
    s = expr_str.strip()
    s = s.replace("^", "**")                  # permitir x^2
    s = s.replace("ln(", "log(")              # permitir ln(x)
    s = s.replace("math.pi", "pi")
    s = re.sub(r"math\.e\b", "E", s)          # solo la constante e
    s = re.sub(r"math\.", "", s)              # quitar prefijo math.

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)
    df_expr = diff(expr, x)

    # Usar módulos math + numpy para mayor compatibilidad
    df_func = lambdify(x, df_expr, modules=["math", "numpy"])
    return df_func, str(df_expr)


def _crear_funcion_segura(f_str):
    """
    Crea una función evaluable de forma relativamente segura para f(x).
    Permite usar math.*, np.*, y funciones básicas.
    """
    expr = f_str.strip().replace("^", "**").replace("ln(", "math.log(")

    allowed_globals = {
        "__builtins__": {},
        "math": math,
        "np": np,
        "abs": abs,
        "pow": pow,
    }

    def f(x):
        allowed_locals = {"x": x}
        return eval(expr, allowed_globals, allowed_locals)

    return f


def metodo_newton_raphson(f, df, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Encuentra la raíz de una función usando el método de Newton-Raphson.

    Error mostrado en tabla:
    - Error absoluto aproximado: |x_{n+1} - x_n|

    Retorna:
    - raíz aproximada de la función
    - historial de iteraciones (lista de x_n)
    """

    print(f"\n{'Iteración':<10} {'x_n':<22} {'f(x_n)':<22} {'f\\'(x_n)':<22} {'Error abs.':<22}")
    print("-" * 110)

    x_actual = x0
    historial_x = [x0]
    iteracion = 0
    error_abs = float("inf")

    while iteracion < max_iteraciones:
        try:
            fx = f(x_actual)
            dfx = df(x_actual)
        except Exception as e:
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None, historial_x

        # Validar valores numéricos
        if not (np.isfinite(fx) and np.isfinite(dfx)):
            print(f"\nError: f(x) o f'(x) no es finito en x = {x_actual}.")
            return None, historial_x

        # Evitar división por cero
        if abs(dfx) < 1e-15:
            print(f"\nError: La derivada es cero (o muy cercana a cero) en x = {x_actual}.")
            return None, historial_x

        # Fórmula de Newton-Raphson
        x_siguiente = x_actual - fx / dfx

        # Error absoluto aproximado entre iteraciones
        error_abs = abs(x_siguiente - x_actual)

        # Imprimir fila (el error corresponde al salto de x_n -> x_{n+1})
        print(f"{iteracion:<10} {x_actual:<22.10f} {fx:<22.10e} {dfx:<22.10e} {error_abs:<22.10e}")

        # Guardar el nuevo valor
        historial_x.append(x_siguiente)

        # Evaluar en el nuevo punto para criterio de parada
        try:
            fx_siguiente = f(x_siguiente)
        except Exception:
            fx_siguiente = float("inf")

        # Criterio de parada:
        # 1) Error absoluto pequeño, o
        # 2) f(x_{n+1}) ya está cerca de 0
        if error_abs < tolerancia or (np.isfinite(fx_siguiente) and abs(fx_siguiente) < tolerancia):
            print(f"\n✓ Raíz encontrada: x = {x_siguiente:.10f}")
            print(f"✓ f({x_siguiente:.10f}) = {fx_siguiente:.10e}")
            print(f"✓ Error absoluto: {error_abs:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente, historial_x

        # Avanzar a la siguiente iteración
        x_actual = x_siguiente
        iteracion += 1

    # Si no convergió dentro del máximo de iteraciones
    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    try:
        fx_final = f(x_actual)
    except Exception:
        fx_final = float("nan")

    print(f"Raíz aproximada: x = {x_actual:.10f}")
    print(f"f({x_actual:.10f}) = {fx_final:.10e}")
    print(f"Error absoluto: {error_abs:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return x_actual, historial_x


def ingresar_funcion():
    """
    Permite al usuario ingresar f(x). La derivada f'(x) se calcula automáticamente.
    """
    print("\n=== MÉTODO DE NEWTON-RAPHSON ===\n")
    print("El método de Newton-Raphson encuentra raíces usando la fórmula:")
    print("  x_{n+1} = x_n - f(x_n) / f'(x_n)\n")
    print("Ingresa la función f(x) en términos de 'x'. La derivada se calcula automáticamente.")
    print("Puedes usar operaciones matemáticas como:")
    print("  - Operadores: +, -, *, /, ** (potencia), ^ (también se acepta)")
    print("  - Funciones: math.sin(), math.cos(), math.tan(), math.exp(), math.log(), math.sqrt()")
    print("  - También: np.sin(), np.cos(), etc.")
    print("  - Ejemplo: x**3 - 2*x - 5")
    print("  - Ejemplo: math.cos(x) - x\n")

    f_str = input("f(x) = ").strip()

    try:
        f = _crear_funcion_segura(f_str)
        f(1)  # prueba rápida
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    try:
        df, df_str = _derivada_simbolica(f_str)
        df(1)  # prueba rápida
        print(f"f'(x) calculada automáticamente: {df_str}\n")
    except Exception as e:
        print(f"Error al calcular la derivada automáticamente: {e}")
        return None, None, None, None

    return f, f_str, df, df_str


def graficar_resultado(f, f_str, df, raiz, x0, historial_x):
    """
    Grafica f(x), marca la raíz y muestra las tangentes de cada iteración.
    """
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
            if np.isfinite(y):
                f_vals.append(y)
            else:
                f_vals.append(np.nan)
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
            plt.plot([x_n, x_n], [0, fx_n], ":", color=colores_tangente[i], linewidth=0.8, alpha=0.5)
        except Exception:
            continue

    try:
        y0 = f(x0)
        if np.isfinite(y0):
            plt.plot(x0, y0, "gs", markersize=10, label=f"Inicio: x₀ = {x0}", zorder=5)
    except Exception:
        pass

    if np.isfinite(f_raiz):
        plt.plot(raiz, f_raiz, "ro", markersize=12, label="Raíz encontrada", zorder=6)

    num_iteraciones = len(historial_x) - 1
    error_final = abs(historial_x[-1] - historial_x[-2]) if len(historial_x) >= 2 else 0.0

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red", linewidth=2)
    anotacion = f"x = {raiz:.8f}\nf(x) = {f_raiz:.2e}\nError abs.: {error_final:.2e}\nIteraciones: {num_iteraciones}"

    f_visible = f_vals[~np.isnan(f_vals)]
    if len(f_visible) > 0:
        y_range = max(float(f_visible.max() - f_visible.min()), 1.0)
    else:
        y_range = 1.0

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
    plt.title(f"Método de Newton-Raphson\nf(x) = {f_str}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.xlim(x_min, x_max)

    if len(f_visible) > 0:
        y_center = (f_visible.max() + f_visible.min()) / 2
        y_half = (f_visible.max() - f_visible.min()) / 2 * 1.3 + 1
        plt.ylim(y_center - y_half, y_center + y_half)

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
        tolerancia_str = input("\nTolerancia (presiona Enter para usar 1e-3): ").strip()
        tolerancia = float(tolerancia_str) if tolerancia_str else 1e-3
        if tolerancia <= 0:
            raise ValueError("La tolerancia debe ser positiva")
    except Exception:
        print("Error en la tolerancia, usando valor por defecto 1e-3")
        tolerancia = 1e-3

    try:
        max_iter_str = input("Número máximo de iteraciones (presiona Enter para usar 100): ").strip()
        max_iteraciones = int(max_iter_str) if max_iter_str else 100
        if max_iteraciones <= 0:
            raise ValueError("Debe ser mayor que cero")
    except Exception:
        print("Error en el número de iteraciones, usando valor por defecto 100")
        max_iteraciones = 100

    print(f"\nResolviendo: f(x) = {f_str}, con x₀ = {x0}")
    raiz, historial_x = metodo_newton_raphson(f, df, x0, tolerancia, max_iteraciones)

    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(f, f_str, df, raiz, x0, historial_x)


if __name__ == "__main__":
    main()