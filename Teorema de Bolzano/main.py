import sys
import os
import math

# Importa utilidades compartidas desde la carpeta raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import crear_funcion_segura, buscar_cambios_de_signo, refinar_cambio, AYUDA_FUNCIONES


def main():
    print("\n=== TEOREMA DE BOLZANO ===\n")
    print("Comprueba si f cambia de signo en un intervalo [a, b].")
    print("Si hay cambio de signo, Bolzano garantiza al menos una raiz en ese tramo.\n")
    print(AYUDA_FUNCIONES)
    print("\nEjemplos:")
    print("  x^3 - 2*x - 5")
    print("  sin(x) - x/2")
    print("  exp(x) - 3*x")
    print("  sqrt(x) - cos(x)")
    print("  ln(x) - x + 2")
    print("  asin(x) - x^2 + 0.5")
    print("  sin(x)*exp(-x) - sqrt(x)/3   <- combina varias funciones\n")

    # Pide la funcion al usuario
    f_str = input("f(x) = ").strip()
    try:
        f = crear_funcion_segura(f_str)
        f(1)  # prueba rapida de sintaxis
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return

    # Pide el intervalo [a, b]
    print("\nIngresa el intervalo [a, b] a analizar:")
    try:
        a = float(input("a = "))
        b = float(input("b = "))
        if a >= b:
            print("Error: 'a' debe ser menor que 'b'")
            return
    except ValueError:
        print("Error: Debes ingresar numeros validos")
        return

    # Numero de subdivisiones: mas subdivisiones -> detecta mas raices cercanas entre si
    try:
        n_str = input("\nNumero de subdivisiones para la busqueda (Enter para 1000): ").strip()
        n = int(n_str) if n_str else 1000
        if n <= 0:
            raise ValueError
    except Exception:
        n = 1000

    # Evalua f en los extremos del intervalo completo
    try:
        fa = f(a)
        fb = f(b)
    except Exception as e:
        print(f"\nError al evaluar f en los extremos: {e}")
        return

    print(f"\nAnalizando f(x) = {f_str}  en  [{a}, {b}]")
    print(f"f({a}) = {fa:.6f}")
    print(f"f({b}) = {fb:.6f}")

    # Comprobacion global de Bolzano en el intervalo completo
    if math.isfinite(fa) and math.isfinite(fb):
        if fa * fb < 0:
            print("\nBolzano en [a, b] completo: SI hay cambio de signo -> existe al menos una raiz.")
        elif fa * fb > 0:
            print("\nBolzano en [a, b] completo: NO hay cambio de signo en los extremos.")
            print("Puede haber raices interiores (numero par) o ninguna. Buscando subintervalos...")
        else:
            # Uno de los extremos es exactamente cero
            if fa == 0:
                print(f"\nf({a}) = 0 exactamente: a = {a} ya es una raiz.")
            if fb == 0:
                print(f"\nf({b}) = 0 exactamente: b = {b} ya es una raiz.")

    # Busca todos los subintervalos con cambio de signo
    cambios = buscar_cambios_de_signo(f, a, b, n)

    print(f"\n{'─'*70}")

    if not cambios:
        print("No se detecto ningun cambio de signo en el intervalo.")
        print("Conclusiones posibles:")
        print("  - f no tiene raices reales en [a, b]")
        print("  - f tiene un numero par de raices muy cercanas (no detectadas con esta subdivision)")
        print("  - f tiene una raiz de multiplicidad par (toca el eje sin cruzarlo)")
        print(f"\nSugerencia: prueba con mas subdivisiones (actualmente {n}).")
        return

    print(f"Se encontraron {len(cambios)} cambio(s) de signo:\n")
    print(f"  {'#':<5} {'Subintervalo':<30} {'f(izq)':<18} {'f(der)':<18} {'Raiz aproximada'}")
    print(f"  {'─'*90}")

    for i, (xi, xd, fi, fd) in enumerate(cambios, 1):
        # Refina la posicion de la raiz dentro del subintervalo
        raiz = refinar_cambio(f, xi, xd)
        print(f"  {i:<5} [{xi:.6f}, {xd:.6f}]   {fi:<18.6f} {fd:<18.6f} x ≈ {raiz:.10f}")

    print(f"\n{'─'*70}")

    if len(cambios) == 1:
        raiz = refinar_cambio(f, cambios[0][0], cambios[0][1])
        print(f"Existe exactamente 1 cambio de signo.")
        print(f"Raiz aproximada: x ≈ {raiz:.10f}")
        print(f"f({raiz:.10f}) = {f(raiz):.6e}")
    else:
        print(f"Existen al menos {len(cambios)} raices reales en [{a}, {b}].")
        print("Usa uno de los metodos numericos con el subintervalo que te interese.")


if __name__ == "__main__":
    main()
