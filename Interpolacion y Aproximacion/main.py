import sys
import os
import math
import numpy as np

# Importa utilidades compartidas desde la carpeta raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import crear_funcion_segura, AYUDA_FUNCIONES


# =============================================================================
# 1. REGRESION LINEAL
#    Modelo: y = b0 + b1*x
#    Formulas de Cramer (minimos cuadrados):
#        b1 = (n*sum(xi*yi) - sum(xi)*sum(yi)) / (n*sum(xi^2) - (sum(xi))^2)
#        b0 = (sum(xi^2)*sum(yi) - sum(xi*yi)*sum(xi)) / (n*sum(xi^2) - (sum(xi))^2)
# =============================================================================

def regresion_lineal(x, y):
    # Sumas necesarias para las formulas de Cramer
    n      = len(x)
    sum_x  = np.sum(x)
    sum_y  = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Denominador comun a b0 y b1
    denominador = n * sum_x2 - sum_x ** 2
    if abs(denominador) < 1e-14:
        raise ValueError("Denominador nulo: los valores de x son constantes o colineales.")

    # Calcula los coeficientes
    b1 = (n * sum_xy - sum_x * sum_y) / denominador
    b0 = (sum_x2 * sum_y - sum_xy * sum_x) / denominador

    # Error cuadratico: suma de residuos al cuadrado
    residuos = y - (b0 + b1 * x)
    ec = float(np.sum(residuos ** 2))

    return float(b0), float(b1), ec


# =============================================================================
# 2. REGRESION POLINOMIAL
#    Modelo: y = b0 + b1*x + b2*x^2 + ... + bm*x^m
#    Sistema normal resuelto con factorizacion QR (numéricamente estable)
# =============================================================================

def regresion_polinomial(x, y, grado):
    n = len(x)
    if grado < 1:
        raise ValueError("El grado debe ser >= 1.")
    if grado >= n:
        raise ValueError(
            f"El grado ({grado}) debe ser menor que el numero de puntos ({n}). "
            "Para grado = n-1 usa interpolacion de Newton o Lagrange."
        )

    # Matriz de Vandermonde: A[i,j] = xi^j  (columna j = potencia j-esima)
    A = np.vander(x, N=grado + 1, increasing=True)

    # Resuelve el sistema normal A^T*A * b = A^T*y via lstsq (mas estable que Cramer directo)
    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # Error cuadratico
    y_pred = A @ coefs
    ec = float(np.sum((y - y_pred) ** 2))

    return coefs, ec


def _evaluar_polinomio(coefs, x_eval):
    """Evalua el polinomio [b0, b1, ..., bm] en x_eval usando el algoritmo de Horner."""
    return np.polyval(np.asarray(coefs)[::-1], np.asarray(x_eval, dtype=float))


# =============================================================================
# 3. REGRESION A FUNCION CONOCIDA (minimos cuadrados generalizado)
#    Modelo: F(x) = a0*phi0(x) + a1*phi1(x) + ... + ar*phir(x)
#    Se construye la matriz de diseno Phi[i,k] = phi_k(xi) y se resuelve
#    el sistema normal Phi^T*Phi * a = Phi^T*y
# =============================================================================

def regresion_funcion_conocida(x, y, funciones_base):
    n = len(x)
    r = len(funciones_base)

    if r == 0:
        raise ValueError("Debe suministrarse al menos una funcion base.")
    if r > n:
        raise ValueError(f"Numero de funciones base ({r}) mayor que numero de datos ({n}).")

    # Construye la matriz de diseno: cada columna es una funcion base evaluada en todos los xi
    Phi = np.column_stack([phi(x) for phi in funciones_base])

    if not np.all(np.isfinite(Phi)):
        raise ValueError("Alguna funcion base produce NaN o Inf en los puntos dados.")

    # Resuelve el sistema normal via lstsq
    coefs, _, rank, _ = np.linalg.lstsq(Phi, y, rcond=None)

    # Verifica que las funciones base sean linealmente independientes
    if rank < r:
        raise ValueError(
            f"Las funciones base son linealmente dependientes (rango={rank}<{r}). "
            "Elimina funciones redundantes."
        )

    # Error cuadratico
    y_pred = Phi @ coefs
    ec = float(np.sum((y - y_pred) ** 2))

    return coefs, ec


# =============================================================================
# 4. INTERPOLACION DE NEWTON (diferencias divididas)
#    P_n(x) = f[x0] + f[x0,x1]*(x-x0) + f[x0,x1,x2]*(x-x0)*(x-x1) + ...
# =============================================================================

def _tabla_diferencias_divididas(x, y):
    """
    Construye la tabla completa de diferencias divididas.
    tabla[i,0] = f(xi)
    tabla[0,k] = coeficientes de Newton f[x0,...,xk]
    """
    n = len(x)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y.copy()   # primera columna: valores f(xi)

    for j in range(1, n):
        for i in range(n - j):
            denom = x[i + j] - x[i]
            tabla[i, j] = (tabla[i + 1, j - 1] - tabla[i, j - 1]) / denom

    return tabla


def interpolacion_newton(x, y, x_eval):
    """
    Calcula el polinomio interpolante de Newton y lo evalua en x_eval.
    Usa el esquema de Horner anidado para mayor eficiencia y estabilidad.

    Algoritmo de Horner para Newton:
        P = coef[n-1]
        para k desde n-2 hasta 0:
            P = coef[k] + (x_eval - x[k]) * P
    """
    tabla = _tabla_diferencias_divididas(x, y)
    coefs = tabla[0, :]   # primera fila: coeficientes f[x0], f[x0,x1], ...

    x_eval = np.asarray(x_eval, dtype=float)
    n = len(coefs)

    # Evaluacion por esquema de Horner
    resultado = np.full_like(x_eval, coefs[n - 1], dtype=float)
    for k in range(n - 2, -1, -1):
        resultado = coefs[k] + (x_eval - x[k]) * resultado

    return resultado, coefs, tabla


# =============================================================================
# 5. INTERPOLACION DE LAGRANGE
#    P_n(x) = sum_i  yi * Li(x)
#    Li(x) = prod_{j!=i} (x - xj) / (xi - xj)
# =============================================================================

def interpolacion_lagrange(x, y, x_eval):
    """
    Calcula el polinomio interpolante de Lagrange en x_eval.
    Usa la formula baricentrica para mayor estabilidad numerica:
        P(x) = [sum_i wi/(x-xi)*yi] / [sum_i wi/(x-xi)]
    donde wi = 1 / prod_{j!=i}(xi - xj)
    """
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    n = len(x)
    resultado = np.zeros_like(x_eval, dtype=float)

    # Calcula los pesos baricentricos: wi = 1 / prod_{j!=i}(xi - xj)
    pesos = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                pesos[i] *= (x[i] - x[j])
    pesos = 1.0 / pesos

    for idx, xp in enumerate(x_eval):
        diferencias = xp - x

        # Caso especial: xp coincide exactamente con un nodo
        nodo_exacto = np.where(diferencias == 0.0)[0]
        if len(nodo_exacto) > 0:
            resultado[idx] = y[nodo_exacto[0]]
            continue

        # Formula baricentrica de Lagrange
        terminos = pesos / diferencias
        resultado[idx] = np.dot(terminos, y) / np.sum(terminos)

    return resultado if resultado.size > 1 else float(resultado[0])


# =============================================================================
# UTILIDADES DE IMPRESION
# =============================================================================

def _imprimir_tabla_datos(x, y):
    """Imprime los puntos (xi, yi) en formato tabla."""
    print(f"\n  {'i':>4}  {'xi':>14}  {'yi':>14}")
    print("  " + "-" * 36)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"  {i:>4}  {xi:>14.6g}  {yi:>14.6g}")


def _imprimir_tabla_diferencias(x, y):
    """Imprime la tabla completa de diferencias divididas de forma legible."""
    tabla = _tabla_diferencias_divididas(x, y)
    n = len(x)
    ancho = 14

    encabezado = f"{'xi':>{ancho}} {'f(xi)':>{ancho}}"
    for k in range(1, n):
        encabezado += f" {'Orden ' + str(k):>{ancho}}"
    print(encabezado)
    print("-" * ancho * (n + 1))

    for i in range(n):
        fila = f"{x[i]:>{ancho}.6g} {tabla[i, 0]:>{ancho}.6g}"
        for j in range(1, n - i):
            fila += f" {tabla[i, j]:>{ancho}.6g}"
        print(fila)

    print("\nCoeficientes de Newton (primera fila):")
    for k, c in enumerate(tabla[0, :]):
        print(f"  f[x0..x{k}] = {c:.8g}")


def _imprimir_resultado(nombre, coefs_etiquetas, ec):
    """Imprime los coeficientes y el error cuadratico de un ajuste."""
    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"{'='*55}")
    for etiq, val in coefs_etiquetas:
        print(f"  {etiq:>8} = {val:+.8g}")
    print(f"\n  Error cuadratico  Ec = {ec:.6e}")
    print(f"{'='*55}")


# =============================================================================
# ENTRADA DE DATOS
# =============================================================================

def ingresar_datos():
    """
    Pide al usuario los puntos (xi, yi).
    Se ingresan como listas de valores separados por espacios.
    Retorna (x, y) como arrays numpy, o (None, None) si hay error.
    """
    print("\nIngresa los puntos de datos.")
    print("Formato: valores separados por espacios  (ej: 1 2 3 4 5)\n")

    try:
        x_str = input("x = ").strip()
        y_str = input("y = ").strip()
        x = np.array([float(v) for v in x_str.split()])
        y = np.array([float(v) for v in y_str.split()])
    except ValueError:
        print("Error: Ingresa solo numeros separados por espacios.")
        return None, None

    # Validaciones basicas
    if len(x) != len(y):
        print(f"Error: x tiene {len(x)} valores e y tiene {len(y)}. Deben ser iguales.")
        return None, None
    if len(x) < 2:
        print("Error: Se necesitan al menos 2 puntos.")
        return None, None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        print("Error: Los valores no deben contener NaN ni Inf.")
        return None, None

    return x, y


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def main():
    print("\n=== INTERPOLACION Y APROXIMACION DE FUNCIONES ===\n")
    print("Metodos disponibles:")
    print("  1. Regresion lineal          y = b0 + b1*x")
    print("  2. Regresion polinomial      y = b0 + b1*x + ... + bm*x^m")
    print("  3. Regresion funcion conocida  F(x) = a0*phi0(x) + a1*phi1(x) + ...")
    print("  4. Interpolacion de Newton   (diferencias divididas)")
    print("  5. Interpolacion de Lagrange")

    try:
        opcion = int(input("\nElige un metodo [1-5]: ").strip())
    except ValueError:
        print("Error: Ingresa un numero entre 1 y 5.")
        return

    if opcion not in range(1, 6):
        print("Error: Opcion no valida. Elige entre 1 y 5.")
        return

    # ── Lectura de datos comun a todos los metodos ────────────────────────
    x, y = ingresar_datos()
    if x is None:
        return

    print("\nPuntos ingresados:")
    _imprimir_tabla_datos(x, y)

    # ── Metodo 1: Regresion lineal ────────────────────────────────────────
    if opcion == 1:
        print("\n--- Regresion lineal:  y = b0 + b1*x ---")

        try:
            b0, b1, ec = regresion_lineal(x, y)
        except ValueError as e:
            print(f"Error: {e}")
            return

        _imprimir_resultado(
            "REGRESION LINEAL   y = b0 + b1*x",
            [("b0", b0), ("b1", b1)],
            ec
        )

        # Evaluacion opcional en un punto
        xp_str = input("\nEvaluar en x (Enter para omitir): ").strip()
        if xp_str:
            try:
                xp = float(xp_str)
                yp = b0 + b1 * xp
                print(f"  F({xp}) = {yp:.10g}")
            except ValueError:
                print("Error: Ingresa un numero valido.")

    # ── Metodo 2: Regresion polinomial ────────────────────────────────────
    elif opcion == 2:
        print("\n--- Regresion polinomial:  y = b0 + b1*x + ... + bm*x^m ---")

        try:
            grado = int(input("Grado del polinomio m = ").strip())
        except ValueError:
            print("Error: Ingresa un entero.")
            return

        try:
            coefs, ec = regresion_polinomial(x, y, grado)
        except ValueError as e:
            print(f"Error: {e}")
            return

        etiquetas = [(f"b{k}", coefs[k]) for k in range(len(coefs))]
        _imprimir_resultado(f"REGRESION POLINOMIAL  grado {grado}", etiquetas, ec)

        xp_str = input("\nEvaluar en x (Enter para omitir): ").strip()
        if xp_str:
            try:
                xp = float(xp_str)
                yp = _evaluar_polinomio(coefs, xp)
                print(f"  P({xp}) = {float(yp):.10g}")
            except ValueError:
                print("Error: Ingresa un numero valido.")

    # ── Metodo 3: Regresion a funcion conocida ────────────────────────────
    elif opcion == 3:
        print("\n--- Regresion funcion conocida:  F(x) = a0*phi0(x) + a1*phi1(x) + ... ---")
        print("\nIngresa cada funcion base como expresion en x (igual que en los otros metodos).")
        print(AYUDA_FUNCIONES)
        print("\nEjemplos de funciones base:")
        print("  1          <- termino constante")
        print("  x")
        print("  x^2")
        print("  sin(x)")
        print("  exp(x)")
        print("  ln(x)      <- requiere x > 0")
        print("  sqrt(x)    <- requiere x >= 0")
        print("  1/x        <- requiere x != 0\n")

        # Pide el numero de funciones base
        try:
            r = int(input("Numero de funciones base r = ").strip())
            if r < 1:
                raise ValueError
        except ValueError:
            print("Error: Ingresa un entero >= 1.")
            return

        # Lee cada funcion base como string y la convierte con crear_funcion_segura
        # np.vectorize permite aplicar la funcion escalar de utils a arrays numpy
        funciones_base = []
        nombres_base   = []
        for k in range(r):
            phi_str = input(f"phi{k}(x) = ").strip()
            try:
                phi_escalar = crear_funcion_segura(phi_str)
                phi_escalar(1)   # prueba en x=1 para detectar errores de sintaxis
                phi_numpy = np.vectorize(phi_escalar)
                phi_numpy(np.array([1.0]))  # prueba con array
            except Exception as e:
                print(f"Error al interpretar phi{k}(x): {e}")
                return
            funciones_base.append(phi_numpy)
            nombres_base.append(phi_str)

        try:
            coefs, ec = regresion_funcion_conocida(x, y, funciones_base)
        except ValueError as e:
            print(f"Error: {e}")
            return

        modelo_str = " + ".join(f"a{k}*({nb})" for k, nb in enumerate(nombres_base))
        etiquetas  = [(f"a{k}", coefs[k]) for k in range(len(coefs))]
        _imprimir_resultado(
            f"REGRESION FUNCION CONOCIDA\n  F(x) = {modelo_str}",
            etiquetas, ec
        )

        xp_str = input("\nEvaluar en x (Enter para omitir): ").strip()
        if xp_str:
            try:
                xp = float(xp_str)
                xp_arr = np.array([xp])
                yp = sum(c * phi(xp_arr) for c, phi in zip(coefs, funciones_base))
                print(f"  F({xp}) = {np.asarray(yp).ravel()[0]:.10g}")
            except Exception as e:
                print(f"  Error al evaluar: {e}")

    # ── Metodo 4: Interpolacion de Newton ─────────────────────────────────
    elif opcion == 4:
        # Los xi deben ser todos distintos para la interpolacion
        if len(np.unique(x)) < len(x):
            print("Error: Los valores de x deben ser todos distintos para interpolacion.")
            return

        print("\n--- Interpolacion de Newton (diferencias divididas) ---")
        print("\nTabla de diferencias divididas:")
        _imprimir_tabla_diferencias(x, y)

        xp_str = input("\nEvaluar en x = ").strip()
        try:
            xp = float(xp_str)
        except ValueError:
            print("Error: Ingresa un numero valido.")
            return

        y_eval, _, _ = interpolacion_newton(x, y, xp)
        print(f"\nResultado:  P({xp}) = {float(y_eval):.10g}")

    # ── Metodo 5: Interpolacion de Lagrange ───────────────────────────────
    elif opcion == 5:
        # Los xi deben ser todos distintos para la interpolacion
        if len(np.unique(x)) < len(x):
            print("Error: Los valores de x deben ser todos distintos para interpolacion.")
            return

        print("\n--- Interpolacion de Lagrange ---")
        print("Li(x) = prod_{j!=i} (x - xj) / (xi - xj)\n")

        xp_str = input("Evaluar en x = ").strip()
        try:
            xp = float(xp_str)
        except ValueError:
            print("Error: Ingresa un numero valido.")
            return

        y_eval = interpolacion_lagrange(x, y, xp)
        print(f"\nResultado:  P({xp}) = {float(y_eval):.10g}")


if __name__ == "__main__":
    main()
