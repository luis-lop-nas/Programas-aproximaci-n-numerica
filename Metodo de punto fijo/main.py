import sys
import os
import math

# Importa utilidades compartidas desde la carpeta raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import crear_funcion_segura, AYUDA_FUNCIONES


def metodo_punto_fijo(f, g, x0, tolerancia=1e-3, max_iteraciones=100):
    # Cabecera de la tabla: incluye f(x_{n+1}) para ver cuanto se acerca a cero
    print(f"\n{'Iter':<6} {'x_n':<22} {'x_{n+1}':<22} {'f(x_{n+1})':<22} {'EN':<15}")
    print("-" * 92)

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

        # Evalua f en el nuevo punto para ver cuanto se aleja de cero
        try:
            fx_siguiente = f(x_siguiente)
        except Exception:
            fx_siguiente = float('nan')

        # EN = error relativo entre el nuevo punto y el actual
        if abs(x_siguiente) > 1e-15:
            EN = abs(x_siguiente - x_actual) / abs(x_siguiente)
        else:
            EN = abs(x_siguiente - x_actual)  # caso especial: raiz cerca de cero

        # Imprime la fila de esta iteracion (sin EN en la primera)
        if iteracion == 0:
            print(f"{iteracion:<6} {x_actual:<22.10f} {x_siguiente:<22.10f} {fx_siguiente:<22.10e} {'---':<15}")
        else:
            print(f"{iteracion:<6} {x_actual:<22.10f} {x_siguiente:<22.10f} {fx_siguiente:<22.10e} {EN:<15.8e}")

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        # No se evalua en la primera iteracion porque no hay comparacion anterior
        if iteracion > 0 and EN < tolerancia:
            print(f"\nRaiz encontrada: x = {x_siguiente:.10f}")
            print(f"f({x_siguiente:.10f}) = {fx_siguiente:.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_siguiente

        x_actual = x_siguiente  # avanza al siguiente punto

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    print(f"Raiz aproximada: x = {x_actual:.10f}")
    try:
        print(f"f({x_actual:.10f}) = {f(x_actual):.10e}")
    except Exception:
        pass
    return x_actual


def ingresar_funciones():
    print("\n=== METODO DE PUNTO FIJO ===\n")
    print("El metodo resuelve f(x) = 0 reescribiendo como x = g(x) e iterando.")
    print("Necesitas ingresar tanto f(x) como g(x).\n")
    print(AYUDA_FUNCIONES)
    print("\nEjemplos:")
    print("  f(x) = x^3 - x - 2        ->  g(x) = (x + 2)^(1/3)")
    print("  f(x) = cos(x) - x         ->  g(x) = cos(x)")
    print("  f(x) = exp(x) - 3*x       ->  g(x) = exp(x) / 3")
    print("  f(x) = sqrt(x) - cos(x)   ->  g(x) = cos(x)^2")
    print("  f(x) = ln(x) - x + 2      ->  g(x) = exp(x - 2)\n")

    # Pide f(x) para poder mostrar f(x_{n+1}) en la tabla
    f_str = input("f(x) = ").strip()
    try:
        f = crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    # Pide g(x): la funcion de iteracion tal que x = g(x)
    g_str = input("g(x) = ").strip()
    try:
        g = crear_funcion_segura(g_str)
        g(1)
        return f, f_str, g, g_str
    except Exception as e:
        print(f"Error al interpretar g(x): {e}")
        return None, None, None, None


def main():
    f, f_str, g, g_str = ingresar_funciones()
    if f is None:
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

    print(f"\nResolviendo f(x) = {f_str}  con  g(x) = {g_str},  x0 = {x0}")
    metodo_punto_fijo(f, g, x0, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
