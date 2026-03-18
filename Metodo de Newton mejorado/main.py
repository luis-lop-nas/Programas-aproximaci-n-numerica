import math
import re


def _normalizar_expr(s: str) -> str:
    # Elimina espacios extremos y convierte ^ a ** (potencia de Python)
    s = s.strip()
    s = s.replace("^", "**")
    return s


def _derivadas_simbolicas(expr_str):
    # Usa sympy para calcular f'(x) y f''(x) de forma simbolica
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

    # Adapta la expresion al formato que entiende sympy
    s = _normalizar_expr(expr_str)
    s = s.replace("ln(", "log(")       # sympy usa log() para logaritmo natural
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bmath\.", "", s)      # elimina prefijos math. para que sympy los reconozca

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    # Calcula primera y segunda derivada simbolica
    df_expr  = diff(expr, x)
    d2f_expr = diff(df_expr, x)

    # Convierte las derivadas simbolicas a funciones numericas usando math
    df_func  = lambdify(x, df_expr,  modules="math")
    d2f_func = lambdify(x, d2f_expr, modules="math")

    return df_func, str(df_expr), d2f_func, str(d2f_expr)


def _crear_funcion_segura(f_str):
    # Reemplaza ^ por ** y ln( por math.log( para compatibilidad
    expr = _normalizar_expr(f_str)
    expr = expr.replace("ln(", "math.log(")

    # Entorno de evaluacion controlado: solo se permiten estas funciones/constantes
    allowed_globals = {
        "__builtins__": {},   # bloquea funciones peligrosas de Python
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

    # Devuelve una funcion f(x) que evalua la expresion del usuario
    def f(x):
        return eval(expr, allowed_globals, {"x": x})

    return f


def metodo_newton_mejorado(f, df, d2f, x0, tolerancia=1e-3, max_iteraciones=100):
    # Cabecera de la tabla de iteraciones
    print("\n{:<6} {:<22} {:<22} {:<22} {:<15}".format(
        "Iter", "x_n", "f(x_n)", "Paso", "EN"
    ))
    print("-" * 95)

    x_actual = x0  # punto de partida

    for iteracion in range(max_iteraciones):
        # Evalua f, f' y f'' en el punto actual
        try:
            fx   = f(x_actual)
            dfx  = df(x_actual)
            d2fx = d2f(x_actual)
        except Exception as e:
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None

        # Comprueba que los valores sean finitos (evita inf o nan)
        if not (math.isfinite(fx) and math.isfinite(dfx) and math.isfinite(d2fx)):
            print(f"\nError: f(x), f'(x) o f''(x) no es finito en x = {x_actual}.")
            return None

        # Denominador del metodo de Halley: (f')^2 - f * f''
        denom = (dfx * dfx) - (fx * d2fx)

        # Si el denominador es cero, el metodo no puede continuar
        if abs(denom) < 1e-15:
            print(f"\nError: Denominador ~ 0 en x = {x_actual}. No se puede continuar.")
            return None

        # Paso de la iteracion: f*f' / ((f')^2 - f*f'')
        paso = (fx * dfx) / denom

        # Formula de Newton mejorado (Halley): x_{n+1} = x_n - f*f' / ((f')^2 - f*f'')
        x_siguiente = x_actual - paso

        # EN = error relativo entre el nuevo punto y el actual
        if abs(x_siguiente) > 1e-15:
            EN = abs(x_siguiente - x_actual) / abs(x_siguiente)
        else:
            EN = abs(x_siguiente - x_actual)  # caso especial: raiz cerca de cero

        # Imprime la fila de esta iteracion
        print("{:<6d} {:<22.10f} {:<22.10e} {:<22.10e} {:<15.8e}".format(
            iteracion, x_actual, fx, paso, EN
        ))

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        if EN < tolerancia:
            fx_sig = f(x_siguiente)
            print(f"\nRaiz encontrada: x = {x_siguiente:.10f}")
            print(f"f({x_siguiente:.10f}) = {fx_sig:.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_siguiente

        x_actual = x_siguiente  # avanza al siguiente punto

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
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

    # Intenta crear la funcion y la prueba en x=1 para detectar errores de sintaxis
    try:
        f = _crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    # Calcula f'(x) y f''(x) automaticamente con sympy y las muestra al usuario
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

    # Punto inicial x0 desde donde arranca el metodo
    try:
        x0 = float(input("x0 = "))
    except ValueError:
        print("Error: Debes ingresar un numero valido")
        return

    # Tolerancia: criterio de parada por error relativo
    try:
        tol_str = input("\nTolerancia (Enter para 1e-3): ").strip()
        tolerancia = float(tol_str) if tol_str else 1e-3
        if tolerancia <= 0:
            raise ValueError
    except Exception:
        tolerancia = 1e-3

    # Limite de iteraciones para evitar bucles infinitos
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
