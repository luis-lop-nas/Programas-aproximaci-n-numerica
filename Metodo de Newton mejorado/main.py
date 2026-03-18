import sys
import os
import math

# Importa utilidades compartidas desde la carpeta raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import crear_funcion_segura, derivadas_simbolicas, AYUDA_FUNCIONES


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
    print("\n=== METODO DE NEWTON MEJORADO (HALLEY) ===\n")
    print("Formula:  x_{n+1} = x_n - (f*f') / ((f')^2 - f*f'')\n")
    print(AYUDA_FUNCIONES)
    print("\nEjemplos:")
    print("  x^3 - 2*x - 5")
    print("  sin(x) - x/2")
    print("  exp(x) - 3*x")
    print("  sqrt(x) - cos(x)")
    print("  ln(x) - x + 2")
    print("  asin(x) - x^2 + 0.5")
    print("  sin(x)*exp(-x) - sqrt(x)/3   <- combina varias funciones\n")

    f_str = input("f(x) = ").strip()

    # Intenta crear la funcion y la prueba en x=1 para detectar errores de sintaxis
    try:
        f = crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    # Calcula f'(x) y f''(x) automaticamente con sympy y las muestra al usuario
    try:
        df, df_str, d2f, d2f_str = derivadas_simbolicas(f_str)
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
