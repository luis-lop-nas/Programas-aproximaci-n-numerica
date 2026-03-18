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


def metodo_secante(f, x0, x1, tolerancia=1e-3, max_iteraciones=100):
    # Cabecera de la tabla de iteraciones
    print(f"\n{'Iter':<6} {'x_{n-1}':<22} {'x_n':<22} {'x_{n+1}':<22} {'f(x_{n+1})':<18} {'EN':<15}")
    print("-" * 110)

    x_ant = x0  # punto anterior (x_{n-1})
    x_act = x1  # punto actual  (x_n)

    for iteracion in range(max_iteraciones):
        # Evalua f en los dos puntos necesarios para la secante
        try:
            f_ant = f(x_ant)
            f_act = f(x_act)
        except Exception as e:
            print(f"\nError al evaluar f(x): {e}")
            return None

        # Comprueba que los valores sean finitos (evita inf o nan)
        if not (math.isfinite(f_ant) and math.isfinite(f_act)):
            print(f"\nError: f(x) no es finito. El metodo diverge.")
            return None

        # Denominador: f(x_n) - f(x_{n-1}). Si es ~0, la secante es casi horizontal
        denom = f_act - f_ant
        if abs(denom) < 1e-15:
            print(f"\nError: f(x_n) - f(x_{{n-1}}) ~ 0 en iteracion {iteracion}. No se puede continuar.")
            return None

        # Formula de la secante: x_{n+1} = x_n - f(x_n)*(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        x_sig = x_act - f_act * (x_act - x_ant) / denom

        # Comprueba que el nuevo punto sea finito (evita divergencia)
        if not math.isfinite(x_sig):
            print(f"\nError: x_{{n+1}} no es finito. El metodo diverge.")
            return None

        # Evalua f en el nuevo punto para mostrarlo en la tabla
        f_sig = f(x_sig)

        # EN = error relativo entre el nuevo punto y el actual
        if abs(x_sig) > 1e-15:
            EN = abs(x_sig - x_act) / abs(x_sig)
        else:
            EN = abs(x_sig - x_act)  # caso especial: raiz cerca de cero

        # Imprime la fila de esta iteracion (sin EN en la primera)
        if iteracion == 0:
            print(f"{iteracion:<6} {x_ant:<22.10f} {x_act:<22.10f} {x_sig:<22.10f} {f_sig:<18.10e} {'---':<15}")
        else:
            print(f"{iteracion:<6} {x_ant:<22.10f} {x_act:<22.10f} {x_sig:<22.10f} {f_sig:<18.10e} {EN:<15.8e}")

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        # No se evalua en la primera iteracion porque no hay comparacion anterior
        if iteracion > 0 and EN < tolerancia:
            print(f"\nRaiz encontrada: x = {x_sig:.10f}")
            print(f"f({x_sig:.10f}) = {f_sig:.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_sig

        # Desplaza los puntos: el actual pasa a ser el anterior
        x_ant = x_act
        x_act = x_sig

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    print(f"Raiz aproximada: x = {x_act:.10f}")
    print(f"f({x_act:.10f}) = {f(x_act):.10e}")
    return x_act


def ingresar_funcion():
    print("\n=== METODO DE LA SECANTE ===\n")
    print("Formula:")
    print("  x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))\n")
    print("No requiere derivada. Necesita dos puntos iniciales x0 y x1.\n")
    print("Ejemplos:")
    print("  x**3 - 2*x - 5")
    print("  sin(x) - x/2")
    print("  exp(x) - 3*x\n")

    f_str = input("f(x) = ").strip()

    # Intenta crear la funcion y la prueba en x=1 para detectar errores de sintaxis
    try:
        f = _crear_funcion_segura(f_str)
        f(1)
        return f, f_str
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None


def main():
    f, f_str = ingresar_funcion()
    if f is None:
        return

    # Pide los dos puntos iniciales necesarios para arrancar el metodo
    print("\nIngresa los dos puntos iniciales:")
    try:
        x0 = float(input("x0 = "))
        x1 = float(input("x1 = "))
    except ValueError:
        print("Error: Debes ingresar numeros validos")
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

    print(f"\nResolviendo: f(x) = {f_str}, con x0 = {x0}, x1 = {x1}")
    metodo_secante(f, x0, x1, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
