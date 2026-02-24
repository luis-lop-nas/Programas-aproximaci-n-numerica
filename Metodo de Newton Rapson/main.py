import math                        # Funciones matemáticas: sin, cos, exp, log, sqrt, etc.
import matplotlib                   # Para detectar el backend gráfico activo
import matplotlib.pyplot as plt     # Para crear y mostrar las gráficas
import numpy as np                  # Para generar arreglos de puntos y operaciones vectoriales


def metodo_newton_raphson(f, df, x0, tolerancia=1e-3, max_iteraciones=100):
    """
    Encuentra la raíz de una función usando el método de Newton-Raphson.

    Parámetros:
    - f: función a evaluar
    - df: derivada de la función
    - x0: valor inicial de la iteración
    - tolerancia: precisión deseada (por defecto 1e-3)
    - max_iteraciones: número máximo de iteraciones (por defecto 100)

    Retorna:
    - raíz aproximada de la función e historial de iteraciones
    """

    # Imprimir encabezado de la tabla con columnas alineadas
    print(f"\n{'Iteración':<12} {'x_n':<25} {'f(x_n)':<25} {'f\'(x_n)':<20} {'Error':<20}")
    print("-" * 105)

    # Inicializar el punto de partida, el historial y el contador
    x_actual = x0
    historial_x = [x0]   # Guarda todos los x_n para dibujar las tangentes en la gráfica
    iteracion = 0
    error = float('inf')  # Error inicial grande para que entre al bucle sin problema

    # Repetir hasta alcanzar el máximo de iteraciones
    while iteracion < max_iteraciones:

        # Evaluar f y f' en el punto actual
        try:
            fx = f(x_actual)
            dfx = df(x_actual)
        except Exception as e:
            # Si alguna evaluación falla (dominio inválido, etc.), abortar
            print(f"\nError al evaluar en x = {x_actual}: {e}")
            return None, historial_x

        # Verificar que f'(x) no sea cero para evitar división por cero
        # (ocurre en puntos de inflexión o máximos/mínimos de f)
        if abs(dfx) < 1e-15:
            print(f"\nError: La derivada es cero en x = {x_actual}. El método no puede continuar.")
            return None, historial_x

        # Fórmula de Newton-Raphson: x_{n+1} = x_n - f(x_n) / f'(x_n)
        # Geométricamente: es la intersección de la tangente en x_n con el eje x
        x_siguiente = x_actual - fx / dfx

        # El error mide cuánto cambió x entre iteraciones consecutivas
        error = abs(x_siguiente - x_actual)

        # Imprimir fila de la tabla: iteración, x_n, f(x_n), f'(x_n) y el error
        print(f"{iteracion:<12} {x_actual:<25.10f} {fx:<25.10e} {dfx:<20.10e} {error:<20.10e}")

        # Guardar el nuevo punto en el historial para la gráfica
        historial_x.append(x_siguiente)

        # Criterio de parada: el error es pequeño O f(x) ya es prácticamente cero
        if error < tolerancia or abs(fx) < tolerancia:
            print(f"\n✓ Raíz encontrada: x = {x_siguiente:.10f}")
            print(f"✓ f({x_siguiente:.10f}) = {f(x_siguiente):.10e}")
            print(f"✓ Error: {error:.10e}")
            print(f"✓ Iteraciones: {iteracion + 1}")
            return x_siguiente, historial_x

        # Avanzar: el siguiente x pasa a ser el x actual
        x_actual = x_siguiente
        iteracion += 1

    # Si se llega aquí, el método no convergió dentro del límite de iteraciones
    print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({max_iteraciones})")
    print(f"Raíz aproximada: x = {x_actual:.10f}")
    print(f"f({x_actual:.10f}) = {f(x_actual):.10e}")
    print(f"Error: {error:.10e}")
    print(f"Iteraciones: {max_iteraciones}")
    return x_actual, historial_x


def ingresar_funcion():
    """
    Permite al usuario ingresar f(x) y su derivada f'(x).
    """
    # Mostrar encabezado y explicación del método
    print("\n=== MÉTODO DE NEWTON-RAPHSON ===\n")
    print("El método de Newton-Raphson encuentra raíces usando la fórmula:")
    print("  x_{n+1} = x_n - f(x_n) / f'(x_n)\n")
    print("Ingresa la función f(x) y su derivada f'(x) en términos de 'x'.")
    print("Puedes usar operaciones matemáticas como:")
    print("  - Operadores: +, -, *, /, ** (potencia)")
    print("  - Funciones: math.sin(), math.cos(), math.tan(), math.exp(), math.log(), math.sqrt()")
    print("  - Ejemplo f(x):  x**3 - 2*x - 5")
    print("  - Ejemplo f'(x): 3*x**2 - 2")
    print("  - Ejemplo f(x):  math.cos(x) - x")
    print("  - Ejemplo f'(x): -math.sin(x) - 1\n")

    # Leer f(x) como texto y convertirla en función ejecutable
    f_str = input("f(x)  = ")

    try:
        f = lambda x: eval(f_str)
        # Probar con x=1 para detectar errores de sintaxis antes de continuar
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return None, None, None, None

    # Leer f'(x) como texto y convertirla en función ejecutable
    df_str = input("f'(x) = ")

    try:
        df = lambda x: eval(df_str)
        # Probar con x=1 para detectar errores de sintaxis antes de continuar
        df(1)
    except Exception as e:
        print(f"Error al interpretar f'(x): {e}")
        return None, None, None, None

    return f, f_str, df, df_str


def graficar_resultado(f, f_str, df, raiz, x0, historial_x):
    """
    Grafica f(x), marca la raíz y muestra las tangentes de cada iteración.

    Parámetros:
    - f: función evaluada
    - f_str: string de f(x) para el título
    - df: derivada de f
    - raiz: valor de x donde está la raíz
    - x0: valor inicial
    - historial_x: lista de todos los x_n generados durante la iteración
    """
    # Calcular el rango de la gráfica para que quepan todos los puntos visitados
    todos_x = historial_x + [raiz]
    rango = max(abs(max(todos_x) - raiz), abs(min(todos_x) - raiz), 1.0)
    margen = rango * 1.4  # Ampliar un 40% extra para que no quede justo en el borde

    x_min = raiz - margen
    x_max = raiz + margen

    # Generar 600 puntos uniformes en el intervalo para trazar la curva suavemente
    x_vals = np.linspace(x_min, x_max, 600)
    f_vals = []

    # Evaluar f en cada punto; si falla guardar NaN para no romper la gráfica
    for x in x_vals:
        try:
            f_vals.append(f(x))
        except:
            f_vals.append(np.nan)

    f_vals = np.array(f_vals)
    # Valor de f en la raíz (idealmente debe ser ≈ 0)
    f_raiz = f(raiz)

    # Crear el lienzo de la figura con tamaño 12x8 pulgadas
    plt.figure(figsize=(12, 8))

    # Trazar la curva f(x) en azul
    plt.plot(x_vals, f_vals, 'b-', linewidth=2, label=f'f(x) = {f_str}')

    # Trazar la línea y = 0 (eje x) para visualizar dónde f cruza cero
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

    # Línea vertical en la raíz para referencia visual
    plt.axvline(x=raiz, color='g', linestyle='--', linewidth=1, alpha=0.3)

    # --- Dibujar las líneas tangentes de cada iteración ---
    # Cada tangente en x_n toca la curva f(x) y cruza el eje x en x_{n+1}
    # Esto ilustra geométricamente cómo Newton-Raphson converge
    # Se limita a 8 tangentes para no saturar la gráfica
    iteraciones_a_mostrar = historial_x[:-1][:8]
    # Paleta de color naranja progresivo: las primeras tangentes son más claras
    colores_tangente = plt.cm.Oranges(np.linspace(0.4, 0.9, len(iteraciones_a_mostrar)))

    for i, x_n in enumerate(iteraciones_a_mostrar):
        try:
            fx_n = f(x_n)
            dfx_n = df(x_n)
            if abs(dfx_n) < 1e-15:
                continue  # Saltar si la derivada es cero en este punto

            # Calcular la recta tangente: y = f(x_n) + f'(x_n) * (x - x_n)
            # Su intersección con y=0 da exactamente x_{n+1}
            x_next = x_n - fx_n / dfx_n
            tang_x = np.array([x_n - margen * 0.4, x_n + margen * 0.4])
            tang_y = fx_n + dfx_n * (tang_x - x_n)

            # Solo la primera tangente lleva etiqueta para no repetir en la leyenda
            label = 'Tangentes' if i == 0 else None
            plt.plot(tang_x, tang_y, '-', color=colores_tangente[i],
                     linewidth=1.2, alpha=0.75, label=label)

            # Círculo naranja en el punto (x_n, f(x_n)) donde se traza la tangente
            plt.plot(x_n, fx_n, 'o', color=colores_tangente[i], markersize=7, zorder=4)

            # Línea vertical punteada desde (x_n, 0) hasta (x_n, f(x_n))
            # Muestra visualmente desde qué punto de la curva sale la tangente
            plt.plot([x_n, x_n], [0, fx_n], ':', color=colores_tangente[i],
                     linewidth=0.8, alpha=0.5)
        except:
            continue

    # Marcar el valor inicial x0 con un cuadrado verde
    try:
        plt.plot(x0, f(x0), 'gs', markersize=10, label=f'Inicio: x₀ = {x0}', zorder=5)
    except:
        pass

    # Marcar la raíz encontrada con un círculo rojo grande
    plt.plot(raiz, f_raiz, 'ro', markersize=12, label='Raíz encontrada', zorder=6)

    # --- Anotación con el resultado numérico ---
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
    anotacion = f'x = {raiz:.8f}\nf(x) = {f_raiz:.2e}'

    # Calcular el desplazamiento vertical para la anotación según la escala real de f
    f_visible = f_vals[~np.isnan(f_vals)]  # Excluir NaN para obtener rango real
    if len(f_visible) > 0:
        y_range = max(f_visible.max() - f_visible.min(), 1)
    else:
        y_range = 1
    offset_y = y_range * 0.15  # La anotación aparece un 15% del rango por encima del punto

    plt.annotate(anotacion,
                xy=(raiz, f_raiz),                          # Punto al que apunta la flecha
                xytext=(raiz, f_raiz + offset_y),           # Posición del texto
                bbox=bbox_props,
                fontsize=11,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Etiquetas, cuadrícula, título y leyenda
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('f(x)', fontsize=12, fontweight='bold')
    plt.title(f'Método de Newton-Raphson\nf(x) = {f_str}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')

    # Fijar el eje horizontal al rango calculado
    plt.xlim(x_min, x_max)

    # Limitar el eje vertical para que valores muy grandes no aplasten la gráfica
    if len(f_visible) > 0:
        y_center = (f_visible.max() + f_visible.min()) / 2
        y_half = (f_visible.max() - f_visible.min()) / 2 * 1.3 + 1
        plt.ylim(y_center - y_half, y_center + y_half)

    plt.tight_layout()  # Ajustar márgenes automáticamente

    # Guardar la imagen en disco
    nombre_archivo = 'grafica_newton_raphson.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada como: {nombre_archivo}")

    # Mostrar la ventana interactiva solo si el backend lo permite (no en servidores sin pantalla)
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except:
        print("(No se pudo abrir la ventana de la gráfica, pero se guardó el archivo)")


def main():
    # Pedir al usuario f(x) y f'(x) y validarlas
    f, f_str, df, df_str = ingresar_funcion()

    # Si alguna función no pudo interpretarse, terminar el programa
    if f is None:
        return

    # Pedir el valor inicial x0 desde el que arrancará la iteración
    print("\nIngresa el valor inicial x₀:")
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

    # Ejecutar el método y obtener la raíz junto con el historial de iteraciones
    print(f"\nResolviendo: f(x) = {f_str}, con x₀ = {x0}")
    raiz, historial_x = metodo_newton_raphson(f, df, x0, tolerancia, max_iteraciones)

    # Si se encontró solución, generar y guardar la gráfica
    if raiz is not None:
        print("\nGenerando gráfica...")
        graficar_resultado(f, f_str, df, raiz, x0, historial_x)


if __name__ == "__main__":
    main()
