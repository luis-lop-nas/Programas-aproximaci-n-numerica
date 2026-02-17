import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def metodo_regla_falsa(funcion, a, b, tolerancia=1e-3, max_iteraciones=100):
    """
    Encuentra la raíz de una función usando el método de regla falsa.

    Parámetros:
    - funcion: función a evaluar (debe ser una función de Python)
    - a: extremo izquierdo del intervalo
    - b: extremo derecho del intervalo
    - tolerancia: precisión deseada (por defecto 1e-3)
    - max_iteraciones: número máximo de iteraciones (por defecto 100)

    Retorna:
    - raíz aproximada de la función
    """

    # Evaluar la función en los extremos
    fa = funcion(a)
    fb = funcion(b)

    # Verificar que haya cambio de signo
    if fa * fb > 0:
        print("Error: La función debe tener signos opuestos en los extremos del intervalo.")
        print(f"f({a}) = {fa}")
        print(f"f({b}) = {fb}")
        return None

    print(f"\n{'Iteración':<12} {'a':<20} {'b':<20} {'c':<20} {'f(c)':<20} {'Error':<20}")
    print("-" * 115)

    iteracion = 0
    c_anterior = a

    while iteracion < max_iteraciones:
        # Calcular el punto usando regla falsa (interpolación lineal)
        c = b - fb * (b - a) / (fb - fa)
        fc = funcion(c)

        # Calcular el error
        if iteracion > 0:
            error = abs(c - c_anterior)
        else:
            error = abs(b - a)

        # Mostrar información de la iteración
        print(f"{iteracion:<12} {a:<20.10f} {b:<20.10f} {c:<20.10f} {fc:<20.10e} {error:<20.10e}")

        # Verificar si se alcanzó la tolerancia
        if abs(fc) < tolerancia or error < tolerancia:
            print(f"\n✓ Raíz encontrada: x = {c:.10f}")
            print(f"✓ f({c:.10f}) = {fc:.10e}")
            print(f"✓ Error: {error:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return c

        # Determinar el nuevo intervalo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_anterior = c
        iteracion += 1

    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    fc = funcion(c)
    print(f"Raíz aproximada: x = {c:.10f}")
    print(f"f({c:.10f}) = {fc:.10e}")
    print(f"Error: {error:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return c


def ingresar_funcion():
    """
    Permite al usuario ingresar una función personalizada.
    """
    print("\n=== MÉTODO DE REGLA FALSA ===\n")
    print("Ingresa tu función en términos de 'x'.")
    print("Puedes usar operaciones matemáticas como:")
    print("  - Operadores: +, -, *, /, ** (potencia)")
    print("  - Funciones: math.sin(), math.cos(), math.tan(), math.exp(), math.log(), math.sqrt()")
    print("  - Ejemplo: x**3 - 2*x - 5")
    print("  - Ejemplo: math.sin(x) - x/2")
    print("  - Ejemplo: math.exp(x) - 3*x\n")

    funcion_str = input("f(x) = ")

    # Crear una función lambda a partir del string
    try:
        funcion = lambda x: eval(funcion_str)
        # Probar la función
        funcion(0)
        return funcion, funcion_str
    except Exception as e:
        print(f"Error al interpretar la función: {e}")
        return None, None


def graficar_resultado(funcion, funcion_str, raiz, a, b):
    """
    Grafica la función y marca el punto donde se encuentra la raíz.

    Parámetros:
    - funcion: función a graficar
    - funcion_str: string de la función para el título
    - raiz: valor de x donde está la raíz
    - a, b: intervalo original de búsqueda
    """
    # Crear un intervalo extendido para graficar
    margen = (b - a) * 0.3
    x_min = a - margen
    x_max = b + margen

    # Generar puntos para la gráfica
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = []

    # Evaluar la función en cada punto
    for x in x_vals:
        try:
            y_vals.append(funcion(x))
        except:
            y_vals.append(np.nan)

    # Calcular f(raiz)
    f_raiz = funcion(raiz)

    # Crear la figura
    plt.figure(figsize=(12, 8))

    # Graficar la función
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {funcion_str}')

    # Línea horizontal en y=0
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

    # Línea vertical en la raíz
    plt.axvline(x=raiz, color='g', linestyle='--', linewidth=1, alpha=0.3)

    # Marcar el intervalo original
    plt.axvline(x=a, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Intervalo inicial [{a:.2f}, {b:.2f}]')
    plt.axvline(x=b, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

    # Marcar el punto de la raíz
    plt.plot(raiz, f_raiz, 'ro', markersize=12, label=f'Raíz encontrada', zorder=5)

    # Añadir anotación con el resultado
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
    anotacion = f'x = {raiz:.10f}\nf(x) = {f_raiz:.2e}'

    # Determinar la posición de la anotación
    y_range = max(y_vals) - min(y_vals) if not np.isnan(max(y_vals)) else 1
    offset_y = y_range * 0.15

    plt.annotate(anotacion,
                xy=(raiz, f_raiz),
                xytext=(raiz, f_raiz + offset_y),
                bbox=bbox_props,
                fontsize=11,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Configurar el gráfico
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('f(x)', fontsize=12, fontweight='bold')
    plt.title(f'Método de Regla Falsa (Falsa Posición)\nf(x) = {funcion_str}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')

    # Ajustar límites
    plt.xlim(x_min, x_max)

    plt.tight_layout()

    # Guardar la gráfica
    nombre_archivo = 'grafica_regla_falsa.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada como: {nombre_archivo}")

    # Intentar mostrar la gráfica solo si el backend es interactivo
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except:
        print("(No se pudo abrir la ventana de la gráfica, pero se guardó el archivo)")


def main():
    # Ingresar la función
    funcion, funcion_str = ingresar_funcion()

    if funcion is None:
        return

    # Ingresar el intervalo
    print("\nIngresa el intervalo [a, b] donde buscar la raíz:")
    try:
        a = float(input("a = "))
        b = float(input("b = "))

        if a >= b:
            print("Error: 'a' debe ser menor que 'b'")
            return
    except ValueError:
        print("Error: Debes ingresar números válidos")
        return

    # Ingresar tolerancia (opcional)
    try:
        tolerancia_str = input("\nTolerancia (presiona Enter para usar 1e-3): ")
        if tolerancia_str.strip():
            tolerancia = float(tolerancia_str)
        else:
            tolerancia = 1e-3
    except ValueError:
        print("Error en la tolerancia, usando valor por defecto")
        tolerancia = 1e-3

    # Ingresar número máximo de iteraciones (opcional)
    try:
        max_iter_str = input("Número máximo de iteraciones (presiona Enter para usar 100): ")
        if max_iter_str.strip():
            max_iteraciones = int(max_iter_str)
        else:
            max_iteraciones = 100
    except ValueError:
        print("Error en el número de iteraciones, usando valor por defecto")
        max_iteraciones = 100

    # Ejecutar el método de regla falsa
    print(f"\nResolviendo: f(x) = {funcion_str} en el intervalo [{a}, {b}]")
    raiz = metodo_regla_falsa(funcion, a, b, tolerancia, max_iteraciones)

    # Graficar el resultado si se encontró una raíz
    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(funcion, funcion_str, raiz, a, b)


if __name__ == "__main__":
    main()
