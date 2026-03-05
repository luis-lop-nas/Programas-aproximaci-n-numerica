import math
import matplotlib.pyplot as plt
import numpy as np

def metodo_biseccion(funcion, a, b, tolerancia=1e-3, max_iteraciones=100):
    """
    Encuentra la raíz de una función usando el método de bisección
    con criterio de parada por ERROR RELATIVO del intervalo.

    error_rel = (b - a) / abs(a + b)
    (si a+b ~ 0, usa error_abs = (b-a)/2)
    """

    fa = funcion(a)
    fb = funcion(b)

    if fa * fb > 0:
        print("Error: La función debe tener signos opuestos en los extremos del intervalo.")
        print(f"f({a}) = {fa}")
        print(f"f({b}) = {fb}")
        return None

    print(f"\n{'Iteración':<12} {'a':<15} {'b':<15} {'c':<15} {'f(c)':<15} {'Err_rel':<15}")
    print("-" * 90)

    iteracion = 0

    while iteracion < max_iteraciones:
        c = (a + b) / 2
        fc = funcion(c)

        # Error relativo del intervalo (forma típica en apuntes)
        denom = abs(a + b)
        if denom > 1e-15:
            error_rel = abs(b - a) / denom
        else:
            # Evitar división por cero si a+b ~ 0
            error_rel = abs(b - a) / 2  # cae a error absoluto

        print(f"{iteracion:<12} {a:<15.8f} {b:<15.8f} {c:<15.8f} {fc:<15.8f} {error_rel:<15.8e}")

        # Criterio de parada:
        # Si quieres SOLO error relativo, elimina "abs(fc) < tolerancia"
        if abs(fc) < tolerancia or error_rel < tolerancia:
            print(f"\n✓ Raíz encontrada: x = {c:.10f}")
            print(f"✓ f({c:.10f}) = {fc:.10e}")
            print(f"✓ Error relativo: {error_rel:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return c

        # Actualizar intervalo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iteracion += 1

    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    c = (a + b) / 2
    fc = funcion(c)

    denom = abs(a + b)
    if denom > 1e-15:
        error_rel = abs(b - a) / denom
    else:
        error_rel = abs(b - a) / 2

    print(f"Raíz aproximada: x = {c:.10f}")
    print(f"f({c:.10f}) = {fc:.10e}")
    print(f"Error relativo: {error_rel:.10e}")
    return c


def ingresar_funcion():
    print("\n=== MÉTODO DE BISECCIÓN ===\n")
    print("Ingresa tu función en términos de 'x'.")
    print("Usa, por ejemplo:")
    print("  x**3 - 2*x - 5")
    print("  math.sin(x) - x/2")
    print("  math.exp(x) - 3*x\n")

    funcion_str = input("f(x) = ")

    try:
        # Eval con entorno controlado (más seguro que eval libre)
        allowed = {"math": math, "__builtins__": {}}
        funcion = lambda x: eval(funcion_str, allowed, {"x": x})
        funcion(0)
        return funcion, funcion_str
    except Exception as e:
        print(f"Error al interpretar la función: {e}")
        return None, None


def graficar_resultado(funcion, funcion_str, raiz, a, b):
    margen = (b - a) * 0.3
    x_min = a - margen
    x_max = b + margen

    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = []

    for x in x_vals:
        try:
            y_vals.append(funcion(x))
        except:
            y_vals.append(np.nan)

    f_raiz = funcion(raiz)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {funcion_str}')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    plt.axvline(x=raiz, color='g', linestyle='--', linewidth=1, alpha=0.3)

    plt.axvline(x=a, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Intervalo inicial [{a:.2f}, {b:.2f}]')
    plt.axvline(x=b, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

    plt.plot(raiz, f_raiz, 'ro', markersize=12, label='Raíz encontrada', zorder=5)

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
    anotacion = f'x = {raiz:.8f}\nf(x) = {f_raiz:.2e}'

    y_clean = [v for v in y_vals if not np.isnan(v)]
    y_range = (max(y_clean) - min(y_clean)) if y_clean else 1.0
    offset_y = y_range * 0.15

    plt.annotate(
        anotacion,
        xy=(raiz, f_raiz),
        xytext=(raiz, f_raiz + offset_y),
        bbox=bbox_props,
        fontsize=11,
        ha='center',
        arrowprops=dict(arrowstyle='->', color='red', lw=2)
    )

    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('f(x)', fontsize=12, fontweight='bold')
    plt.title(f'Método de Bisección (error relativo)\nf(x) = {funcion_str}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(x_min, x_max)
    plt.tight_layout()

    nombre_archivo = 'grafica_biseccion.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada como: {nombre_archivo}")

    try:
        plt.show()
    except:
        print("(No se pudo abrir la ventana de la gráfica, pero se guardó el archivo)")


def main():
    funcion, funcion_str = ingresar_funcion()
    if funcion is None:
        return

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

    try:
        tolerancia_str = input("\nTolerancia (presiona Enter para usar 1e-3): ")
        tolerancia = float(tolerancia_str) if tolerancia_str.strip() else 1e-3
    except ValueError:
        print("Error en la tolerancia, usando valor por defecto")
        tolerancia = 1e-3

    try:
        max_iter_str = input("Número máximo de iteraciones (presiona Enter para usar 100): ")
        max_iteraciones = int(max_iter_str) if max_iter_str.strip() else 100
    except ValueError:
        print("Error en el número de iteraciones, usando valor por defecto")
        max_iteraciones = 100

    print(f"\nResolviendo: f(x) = {funcion_str} en el intervalo [{a}, {b}]")
    raiz = metodo_biseccion(funcion, a, b, tolerancia, max_iteraciones)

    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(funcion, funcion_str, raiz, a, b)


if __name__ == "__main__":
    main()