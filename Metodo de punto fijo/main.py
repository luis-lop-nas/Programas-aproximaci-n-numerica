import math                        # Funciones matemáticas: sin, cos, exp, log, sqrt, etc.
import matplotlib                   # Para detectar el backend gráfico activo
import matplotlib.pyplot as plt     # Para crear y mostrar las gráficas
import numpy as np                  # Para generar arreglos de puntos y operaciones vectoriales


def metodo_punto_fijo(g, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Encuentra el punto fijo de g(x) (es decir, la raíz de f(x) = g(x) - x = 0)
    usando el método de iteración de punto fijo.

    Parámetros:
    - g: función de iteración g(x), donde el punto fijo cumple g(x) = x
    - x0: valor inicial de la iteración
    - tolerancia: precisión deseada (por defecto 1e-3)
    - max_iteraciones: número máximo de iteraciones (por defecto 100)

    Retorna:
    - punto fijo aproximado (raíz de f(x) = g(x) - x)
    """

    # Imprimir encabezado de la tabla de resultados con columnas alineadas
    print(f"\n{'Iteración':<12} {'x_n':<25} {'g(x_n)':<25} {'Error':<20}")
    print("-" * 85)

    # Inicializar el punto de partida y el contador de iteraciones
    x_actual = x0
    iteracion = 0

    # Repetir hasta alcanzar el máximo de iteraciones permitidas
    while iteracion < max_iteraciones:

        # Evaluar g en el punto actual para obtener el siguiente candidato
        try:
            x_siguiente = g(x_actual)
        except Exception as e:
            # Si la función falla (dominio inválido, etc.), abortar
            print(f"\nError al evaluar g({x_actual}): {e}")
            return None

        # El error es la distancia entre la iteración actual y la anterior
        # Mide cuánto se movió x: si es pequeño, el método convergió
        error = abs(x_siguiente - x_actual)

        # Mostrar fila de la tabla: iteración, x_n, g(x_n) y el error
        print(f"{iteracion:<12} {x_actual:<25.10f} {x_siguiente:<25.10f} {error:<20.10e}")

        # Criterio de parada: el error cayó por debajo de la tolerancia pedida
        if error < tolerancia:
            print(f"\n✓ Punto fijo encontrado: x = {x_siguiente:.10f}")
            print(f"✓ g({x_siguiente:.10f}) = {g(x_siguiente):.10f}")
            print(f"✓ Error: {error:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente

        # Preparar la siguiente iteración: el nuevo x_n pasa a ser x_{n+1}
        x_actual = x_siguiente
        iteracion += 1

    # Si se llega aquí, el método no convergió dentro del límite de iteraciones
    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    print(f"Punto fijo aproximado: x = {x_actual:.10f}")
    print(f"g({x_actual:.10f}) = {g(x_actual):.10f}")
    print(f"Error: {error:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return x_actual


def ingresar_funcion():
    """
    Permite al usuario ingresar la función de iteración g(x).
    """
    # Mostrar encabezado y explicación del método
    print("\n=== MÉTODO DE PUNTO FIJO ===\n")
    print("El método de punto fijo resuelve f(x) = 0 reescribiendo la ecuación")
    print("como x = g(x) y aplicando la iteración x_{n+1} = g(x_n).\n")
    print("Ingresa la función de iteración g(x) en términos de 'x'.")
    print("Puedes usar operaciones matemáticas como:")
    print("  - Operadores: +, -, *, /, ** (potencia)")
    print("  - Funciones: math.sin(), math.cos(), math.tan(), math.exp(), math.log(), math.sqrt()")
    print("  - Ejemplo: (x**2 + 2) / 3           [para f(x) = x^2 - 3x + 2 = 0]")
    print("  - Ejemplo: math.sqrt(2*x + 3)        [para f(x) = x^2 - 2x - 3 = 0]")
    print("  - Ejemplo: math.exp(-x)              [para f(x) = x - e^(-x) = 0]")
    print("  - Ejemplo: (math.exp(x) + 2) / 3    [para f(x) = e^x - 3x + 2 = 0]\n")

    # Leer la expresión de g(x) como texto
    g_str = input("g(x) = ")

    # Convertir el texto en una función ejecutable usando eval
    # eval interpreta el string como código Python en cada llamada
    try:
        g = lambda x: eval(g_str)
        # Prueba con x=1 para detectar errores de sintaxis antes de usarla
        g(1)
        return g, g_str
    except Exception as e:
        print(f"Error al interpretar la función: {e}")
        return None, None


def graficar_resultado(g, g_str, punto_fijo, x0, historial_x):
    """
    Grafica g(x), la línea y=x, el punto fijo y el diagrama de telaraña
    que muestra la convergencia del método.

    Parámetros:
    - g: función de iteración
    - g_str: string de g(x) para el título
    - punto_fijo: valor donde g(x) = x
    - x0: valor inicial
    - historial_x: lista de todos los x_n generados durante la iteración
    """
    # Calcular el rango de la gráfica para que quepan todos los puntos visitados
    todos_x = historial_x + [punto_fijo]
    x_centro = punto_fijo
    # El rango cubre la distancia máxima desde el punto fijo a cualquier iteración
    rango = max(abs(max(todos_x) - punto_fijo), abs(min(todos_x) - punto_fijo), 1.0)
    margen = rango * 1.5  # Ampliar un 50% extra para que no quede justo en el borde

    x_min = x_centro - margen
    x_max = x_centro + margen

    # Generar 600 puntos uniformes en el intervalo para trazar las curvas suavemente
    x_vals = np.linspace(x_min, x_max, 600)
    g_vals = []

    # Evaluar g en cada punto; si falla (raíz negativa, etc.) guardar NaN para no romper la gráfica
    for x in x_vals:
        try:
            g_vals.append(g(x))
        except:
            g_vals.append(np.nan)

    g_vals = np.array(g_vals)

    # Valor de g en el punto fijo (debe ser igual al propio punto fijo)
    g_punto_fijo = g(punto_fijo)

    # Crear el lienzo de la figura con tamaño 12x8 pulgadas
    plt.figure(figsize=(12, 8))

    # Trazar la curva g(x) en azul
    plt.plot(x_vals, g_vals, 'b-', linewidth=2, label=f'g(x) = {g_str}')

    # Trazar la bisectriz y = x en negro punteado
    # El punto fijo es la intersección entre g(x) y esta recta
    plt.plot(x_vals, x_vals, 'k--', linewidth=1.5, label='y = x', alpha=0.6)

    # --- Diagrama de telaraña (cobweb) ---
    # Visualiza cómo las iteraciones se van acercando al punto fijo:
    #   - Línea vertical: sube desde (x_n, x_n) hasta (x_n, g(x_n))  -> evalúa g
    #   - Línea horizontal: va desde (x_n, g(x_n)) hasta (g(x_n), g(x_n)) -> proyecta sobre y=x
    if len(historial_x) > 1:
        cobweb_x = []
        cobweb_y = []
        x_n = historial_x[0]
        # Empezar desde el punto inicial sobre la bisectriz
        cobweb_x.append(x_n)
        cobweb_y.append(x_n)
        for x_next in historial_x[1:]:
            # Subir hasta la curva g(x): tramo vertical
            cobweb_x.append(x_n)
            cobweb_y.append(x_next)
            # Moverse horizontalmente a la bisectriz: tramo horizontal
            cobweb_x.append(x_next)
            cobweb_y.append(x_next)
            x_n = x_next
        plt.plot(cobweb_x, cobweb_y, 'g-', linewidth=1, alpha=0.5, label='Iteraciones (telaraña)')

    # Marcar el punto inicial x0 sobre la bisectriz (cuadrado verde)
    try:
        g_x0 = g(x0)
        plt.plot(x0, x0, 'gs', markersize=10, label=f'Inicio: x₀ = {x0}', zorder=5)
    except:
        pass

    # Marcar el punto fijo encontrado con un círculo rojo
    plt.plot(punto_fijo, g_punto_fijo, 'ro', markersize=12, label='Punto fijo', zorder=6)

    # --- Anotación con el resultado numérico ---
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
    anotacion = f'x = {punto_fijo:.8f}\ng(x) = {g_punto_fijo:.8f}'

    # Calcular el desplazamiento vertical de la anotación según la escala de la gráfica
    g_visible = g_vals[~np.isnan(g_vals)]  # Ignorar los NaN para calcular el rango real
    if len(g_visible) > 0:
        y_range = max(g_visible.max() - g_visible.min(), abs(x_max - x_min), 1)
    else:
        y_range = abs(x_max - x_min)
    offset_y = y_range * 0.12  # La anotación aparece un 12% del rango por encima del punto

    plt.annotate(anotacion,
                xy=(punto_fijo, g_punto_fijo),           # Punto al que apunta la flecha
                xytext=(punto_fijo, g_punto_fijo + offset_y),  # Posición del texto
                bbox=bbox_props,
                fontsize=11,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Etiquetas, cuadrícula, título y leyenda
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('g(x)', fontsize=12, fontweight='bold')
    plt.title(f'Método de Punto Fijo\ng(x) = {g_str}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')

    # Fijar el eje horizontal al rango calculado
    plt.xlim(x_min, x_max)

    # Limitar el eje vertical para que valores muy grandes no aplasten la gráfica
    if len(g_visible) > 0:
        y_center = (g_visible.max() + g_visible.min()) / 2
        y_half = (g_visible.max() - g_visible.min()) / 2 * 1.3 + 1
        plt.ylim(y_center - y_half, y_center + y_half)

    plt.tight_layout()  # Ajustar márgenes automáticamente

    # Guardar la imagen en disco
    nombre_archivo = 'grafica_punto_fijo.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada como: {nombre_archivo}")

    # Mostrar la ventana interactiva solo si el backend lo permite (no en servidores sin pantalla)
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except:
        print("(No se pudo abrir la ventana de la gráfica, pero se guardó el archivo)")


def metodo_punto_fijo_con_historial(g, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Versión interna del método que también devuelve el historial de iteraciones
    para poder dibujar el diagrama de telaraña en la gráfica.
    """
    # Guardar todos los x_n visitados, empezando por el valor inicial
    historial_x = [x0]

    # Encabezado de la tabla
    print(f"\n{'Iteración':<12} {'x_n':<25} {'g(x_n)':<25} {'Error':<20}")
    print("-" * 85)

    x_actual = x0
    iteracion = 0

    while iteracion < max_iteraciones:

        # Aplicar la iteración: x_{n+1} = g(x_n)
        try:
            x_siguiente = g(x_actual)
        except Exception as e:
            print(f"\nError al evaluar g({x_actual}): {e}")
            return None, historial_x

        # Diferencia absoluta entre iteraciones consecutivas
        error = abs(x_siguiente - x_actual)

        # Imprimir la fila de esta iteración
        print(f"{iteracion:<12} {x_actual:<25.10f} {x_siguiente:<25.10f} {error:<20.10e}")

        # Agregar el nuevo punto al historial para la gráfica
        historial_x.append(x_siguiente)

        # Verificar convergencia
        if error < tolerancia:
            print(f"\n✓ Punto fijo encontrado: x = {x_siguiente:.10f}")
            print(f"✓ g({x_siguiente:.10f}) = {g(x_siguiente):.10f}")
            print(f"✓ Error: {error:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente, historial_x

        # Avanzar a la siguiente iteración
        x_actual = x_siguiente
        iteracion += 1

    # Se agotaron las iteraciones sin converger
    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    print(f"Punto fijo aproximado: x = {x_actual:.10f}")
    print(f"g({x_actual:.10f}) = {g(x_actual):.10f}")
    print(f"Error: {error:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return x_actual, historial_x


def main():
    # Pedir al usuario que ingrese g(x) y validarla
    g, g_str = ingresar_funcion()

    # Si la función no pudo interpretarse, terminar el programa
    if g is None:
        return

    # Pedir el valor inicial x0 desde el que arrancará la iteración
    print("\nIngresa el valor inicial x₀ para la iteración:")
    try:
        x0 = float(input("x0 = "))
    except ValueError:
        print("Error: Debes ingresar un número válido")
        return

    # Pedir la tolerancia; si el usuario presiona Enter se usa el valor por defecto 1e-3
    try:
        tolerancia_str = input("\nTolerancia (presiona Enter para usar 1e-3): ")
        if tolerancia_str.strip():
            tolerancia = float(tolerancia_str)
        else:
            tolerancia = 1e-3
    except ValueError:
        print("Error en la tolerancia, usando valor por defecto")
        tolerancia = 1e-3

    # Pedir el máximo de iteraciones; si el usuario presiona Enter se usan 100
    try:
        max_iter_str = input("Número máximo de iteraciones (presiona Enter para usar 100): ")
        if max_iter_str.strip():
            max_iteraciones = int(max_iter_str)
        else:
            max_iteraciones = 100
    except ValueError:
        print("Error en el número de iteraciones, usando valor por defecto")
        max_iteraciones = 100

    # Ejecutar el método y obtener el punto fijo junto con el historial de iteraciones
    print(f"\nResolviendo: g(x) = {g_str}, con x₀ = {x0}")
    punto_fijo, historial_x = metodo_punto_fijo_con_historial(g, x0, tolerancia, max_iteraciones)

    # Si se encontró solución, generar y guardar la gráfica
    if punto_fijo is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(g, g_str, punto_fijo, x0, historial_x)


if __name__ == "__main__":
    main()
