import math


def _crear_funcion_segura(f_str):
    # Reemplaza ^ por ** y ln( por math.log( para compatibilidad
    expr = f_str.strip().replace("^", "**").replace("ln(", "math.log(")

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


def metodo_punto_fijo(g, x0, tolerancia=1e-3, max_iteraciones=100):
    # Cabecera de la tabla de iteraciones
    print(f"\n{'Iter':<6} {'x_n':<22} {'x_{n+1}':<22} {'EN':<15}")
    print("-" * 70)

    x_actual = x0  # punto de partida

    for iteracion in range(max_iteraciones):
        # Aplica la funcion de iteracion: x_{n+1} = g(x_n)
        try:
            x_siguiente = g(x_actual)
        except Exception as e:
            print(f"\nError al evaluar g(x) en x = {x_actual}: {e}")
            return None

        # Comprueba que el resultado sea finito (divergencia produce inf o nan)
        if not math.isfinite(x_siguiente):
            print(f"\nError: g(x) no es finito en x = {x_actual}. El metodo diverge.")
            return None

        # EN = error relativo entre el nuevo punto y el actual
        if abs(x_siguiente) > 1e-15:
            EN = abs(x_siguiente - x_actual) / abs(x_siguiente)
        else:
            EN = abs(x_siguiente - x_actual)  # caso especial: raiz cerca de cero

        # Imprime la fila de esta iteracion (sin EN en la primera)
        if iteracion == 0:
            print(f"{iteracion:<6} {x_actual:<22.10f} {x_siguiente:<22.10f} {'---':<15}")
        else:
            print(f"{iteracion:<6} {x_actual:<22.10f} {x_siguiente:<22.10f} {EN:<15.8e}")

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        # No se evalua en la primera iteracion porque no hay comparacion anterior
        if iteracion > 0 and EN < tolerancia:
            print(f"\nRaiz encontrada: x = {x_siguiente:.10f}")
            print(f"g({x_siguiente:.10f}) = {g(x_siguiente):.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_siguiente

        x_actual = x_siguiente  # avanza al siguiente punto

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    print(f"Raiz aproximada: x = {x_actual:.10f}")
    return x_actual


def ingresar_funcion():
    print("\n=== METODO DE PUNTO FIJO ===\n")
    print("El metodo resuelve f(x) = 0 reescribiendo como x = g(x) e iterando.")
    print("Debes ingresar g(x) tal que x_{n+1} = g(x_n).\n")
    print("Ejemplos:")
    print("  Si f(x) = x^3 - x - 2 = 0  ->  g(x) = (x + 2)**(1/3)")
    print("  Si f(x) = cos(x) - x = 0   ->  g(x) = cos(x)")
    print("  Si f(x) = exp(x) - 3*x = 0 ->  g(x) = exp(x) / 3\n")

    g_str = input("g(x) = ").strip()

    # Intenta crear la funcion g y la prueba en x=1 para detectar errores de sintaxis
    try:
        g = _crear_funcion_segura(g_str)
        g(1)
        return g, g_str
    except Exception as e:
        print(f"Error al interpretar g(x): {e}")
        return None, None


def main():
    g, g_str = ingresar_funcion()
    if g is None:
        return

    # Punto inicial x0 desde donde arranca el metodo
    try:
        x0 = float(input("\nx0 = "))
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

    print(f"\nResolviendo con g(x) = {g_str}, x0 = {x0}")
    metodo_punto_fijo(g, x0, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
