import math
import re


def _normalizar_expr(s: str) -> str:
    s = s.strip()
    s = s.replace("^", "**")
    return s


def _derivadas_simbolicas(expr_str):
    try:
        from sympy import symbols, diff, lambdify
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except Exception as e:
        raise RuntimeError("Falta sympy. Instalalo con: pip install sympy") from e

    x = symbols("x")
    s = _normalizar_expr(expr_str)
    s = s.replace("ln(", "log(")
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bmath\.", "", s)

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    df_expr = diff(expr, x)
    d2f_expr = diff(df_expr, x)

    df_func = lambdify(x, df_expr, modules="math")
    d2f_func = lambdify(x, d2f_expr, modules="math")

    return df_func, str(df_expr), d2f_func, str(d2f_expr)


def _crear_funcion_segura(f_str):
    expr = _normalizar_expr(f_str)
    expr = expr.replace("ln(", "math.log(")

    allowed_globals = {
        "__builtins__": {},
        "math": math,
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


def metodo_newton_mejorado(f, df, d2f, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    x_{n+1} = x_n - (f * f') / ((f')^2 - f * f'')
    EN = |x_{n+1} - x_n| / |x_{n+1}|
    """

    print("\n{:<6} {:<22} {:<22} {:<22} {:<15}".format(
        "Iter", "x_n", "f(x_n)", "Paso", "EN"
    ))
    print("-" * 95)

    x_actual = x0

    for iteracion in range(max_iteraciones):
        try:
            fx = f(x_actual)
            dfx = df(x_actual)
            d2fx = d2f(x_actual)
        except Exception as e:
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None

        if not (math.isfinite(fx) and math.isfinite(dfx) and math.isfinite(d2fx)):
            print(f"\nError: f(x), f'(x) o f''(x) no es finito en x = {x_actual}.")
            return None

        denom = (dfx * dfx) - (fx * d2fx)
        if abs(denom) < 1e-15:
            print(f"\nError: Denominador ~ 0 en x = {x_actual}. No se puede continuar.")
            return None

        paso = (fx * dfx) / denom
        x_siguiente = x_actual - paso

        if abs(x_siguiente) > 1e-15:
            EN = abs(x_siguiente - x_actual) / abs(x_siguiente)
        else:
            EN = abs(x_siguiente - x_actual)

        print("{:<6d} {:<22.10f} {:<22.10e} {:<22.10e} {:<15.8e}".format(
            iteracion, x_actual, fx, paso, EN
        ))

        if EN < tolerancia:
            fx_sig = f(x_siguiente)
            print(f"\nRaiz encontrada: x = {x_siguiente:.10f}")
            print(f"f({x_siguiente:.10f}) = {fx_sig:.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_siguiente

        x_actual = x_siguiente

    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    fx_final = f(x_actual)
    print(f"Raiz aproximada: x = {x_actual:.10f}")
    print(f"f({x_actual:.10f}) = {fx_final:.10e}")
    return x_actual


def ingresar_funcion():
    print("\n=== METODO DE NEWTON MEJORADO ===\n")
    print("Formula:")
    print("  x_{n+1} = x_n - (f(x_n) * f'(x_n)) / ((f'(x_n))^2 - f(x_n) * f''(x_n))\n")
    print("Puedes escribir: sin(x), cos(x), exp(x), log(x), sqrt(x), ln(x), pi, e, ^\n")

    f_str = input("f(x) = ").strip()

    try:
        f = _crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    try:
        df, df_str, d2f, d2f_str = _derivadas_simbolicas(f_str)
        df(1); d2f(1)
        print(f"f'(x)  = {df_str}")
        print(f"f''(x) = {d2f_str}\n")
    except Exception as e:
        print(f"Error al calcular derivadas: {e}")
        return None, None, None, None

    return f, f_str, df, d2f


def main():
    f, f_str, df, d2f = ingresar_funcion()
    if f is None:
        return

    try:
        x0 = float(input("x0 = "))
    except ValueError:
        print("Error: Debes ingresar un numero valido")
        return

    try:
        tol_str = input("\nTolerancia (Enter para 1e-3): ").strip()
        tolerancia = float(tol_str) if tol_str else 1e-3
        if tolerancia <= 0:
            raise ValueError
    except Exception:
        tolerancia = 1e-3

    try:
        iter_str = input("Numero maximo de iteraciones (Enter para 100): ").strip()
        max_iteraciones = int(iter_str) if iter_str else 100
        if max_iteraciones <= 0:
            raise ValueError
    except Exception:
        max_iteraciones = 100

    print(f"\nResolviendo: f(x) = {f_str}, con x0 = {x0}")
    metodo_newton_mejorado(f, df, d2f, x0, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
