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


def metodo_biseccion(funcion, a, b, tolerancia=1e-3, max_iteraciones=100):
    # Evalua la funcion en los extremos del intervalo inicial
    fa = funcion(a)
    fb = funcion(b)

    # Condicion de Bolzano: debe haber cambio de signo para garantizar una raiz
    if fa * fb > 0:
        print("Error: La función debe tener signos opuestos en los extremos del intervalo.")
        print(f"f({a}) = {fa}")
        print(f"f({b}) = {fb}")
        return None

    # Cabecera de la tabla de iteraciones
    print(f"\n{'Iter':<6} {'a':<18} {'b':<18} {'c':<18} {'f(c)':<18} {'EN':<15}")
    print("-" * 95)

    iteracion = 0
    c_anterior = None  # guarda el punto medio de la iteracion anterior para calcular EN

    while iteracion < max_iteraciones:
        # Calcula el punto medio del intervalo actual
        c = (a + b) / 2
        fc = funcion(c)

        # EN = error relativo entre el punto medio actual y el anterior
        # En la primera iteracion no hay punto anterior, se omite
        if c_anterior is not None and abs(c) > 1e-15:
            EN = abs(c - c_anterior) / abs(c)
        else:
            EN = float('inf')

        # Imprime la fila de esta iteracion
        if c_anterior is None:
            print(f"{iteracion:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {'---':<15}")
        else:
            print(f"{iteracion:<6} {a:<18.10f} {b:<18.10f} {c:<18.10f} {fc:<18.10e} {EN:<15.8e}")

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        if c_anterior is not None and EN < tolerancia:
            print(f"\nRaiz encontrada: x = {c:.10f}")
            print(f"f({c:.10f}) = {fc:.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return c

        # Actualiza el intervalo conservando el subintervalo donde hay cambio de signo
        if fa * fc < 0:
            # La raiz esta en [a, c] -> el nuevo extremo derecho es c
            b = c
            fb = fc
        else:
            # La raiz esta en [c, b] -> el nuevo extremo izquierdo es c
            a = c
            fa = fc

        c_anterior = c  # guarda el punto medio actual para la siguiente iteracion
        iteracion += 1

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    c = (a + b) / 2
    fc = funcion(c)
    EN = abs(c - c_anterior) / abs(c) if c_anterior is not None and abs(c) > 1e-15 else float('inf')

    print(f"Raiz aproximada: x = {c:.10f}")
    print(f"f({c:.10f}) = {fc:.10e}")
    print(f"EN = {EN:.10e}")
    return c


def ingresar_funcion():
    print("\n=== METODO DE BISECCION ===\n")
    print("Ingresa tu funcion en terminos de 'x'. Ejemplos:")
    print("  x**3 - 2*x - 5")
    print("  sin(x) - x/2")
    print("  exp(x) - 3*x\n")

    funcion_str = input("f(x) = ")

    # Intenta crear la funcion y la prueba en x=0 para detectar errores de sintaxis
    try:
        funcion = _crear_funcion_segura(funcion_str)
        funcion(0)
        return funcion, funcion_str
    except Exception as e:
        print(f"Error al interpretar la funcion: {e}")
        return None, None


def main():
    funcion, funcion_str = ingresar_funcion()
    if funcion is None:
        return

    # Pide el intervalo [a, b] donde se buscara la raiz
    print("\nIngresa el intervalo [a, b] donde buscar la raiz:")
    try:
        a = float(input("a = "))
        b = float(input("b = "))
        if a >= b:
            print("Error: 'a' debe ser menor que 'b'")
            return
    except ValueError:
        print("Error: Debes ingresar numeros validos")
        return

    # Tolerancia: criterio de parada por error relativo
    try:
        tol_str = input("\nTolerancia (Enter para 1e-3): ")
        tolerancia = float(tol_str) if tol_str.strip() else 1e-3
    except ValueError:
        tolerancia = 1e-3

    # Limite de iteraciones para evitar bucles infinitos
    try:
        iter_str = input("Numero maximo de iteraciones (Enter para 100): ")
        max_iteraciones = int(iter_str) if iter_str.strip() else 100
    except ValueError:
        max_iteraciones = 100

    print(f"\nResolviendo: f(x) = {funcion_str}  en  [{a}, {b}]")
    metodo_biseccion(funcion, a, b, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
