import numpy as np


TOL = 1e-12


def leer_arreglo(nombre):
    """Lee una lista de floats separados por espacios."""
    texto = input(f"{nombre} = ").strip()
    try:
        valores = np.array([float(v) for v in texto.split()], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Error en {nombre}: ingresa solo numeros separados por espacios.") from exc

    if valores.size == 0:
        raise ValueError(f"Error: {nombre} no puede estar vacio.")
    if not np.all(np.isfinite(valores)):
        raise ValueError(f"Error: {nombre} contiene NaN o Inf.")
    return valores


def leer_opcion(mensaje, minimo, maximo):
    """Lee una opcion entera en [minimo, maximo]."""
    while True:
        try:
            opcion = int(input(mensaje).strip())
        except ValueError:
            print(f"  Opcion invalida. Ingresa un entero entre {minimo} y {maximo}.")
            continue

        if minimo <= opcion <= maximo:
            return opcion

        print(f"  Opcion fuera de rango. Debe estar entre {minimo} y {maximo}.")


def validar_y_ordenar_puntos(x, y):
    """Valida puntos y los ordena por x ascendente."""
    if x.size != y.size:
        raise ValueError(
            f"Error: x tiene {x.size} valores e y tiene {y.size}. Deben coincidir."
        )
    if x.size < 3:
        raise ValueError("Error: se necesitan al menos 3 puntos para derivadas adelante/central/atras.")

    orden = np.argsort(x)
    x_ord = x[orden]
    y_ord = y[orden]

    if np.any(np.isclose(np.diff(x_ord), 0.0, atol=TOL, rtol=0.0)):
        raise ValueError("Error: los valores de x deben ser distintos.")

    return x_ord, y_ord


def calcular_h(x):
    """Calcula h_i = x_{i+1} - x_i y detecta si h es constante."""
    h = np.diff(x)
    h0 = float(h[0])
    es_constante = np.allclose(h, h0, atol=1e-10, rtol=1e-8)
    error_max = float(np.max(np.abs(h - h0)))
    return h, h0, es_constante, error_max


def elegir_modo_h():
    print("\nModo de trabajo para h:")
    print("  1. Detectar automaticamente si h es constante")
    print("  2. Forzar h constante")
    print("  3. Forzar h no constante")
    return leer_opcion("Elige [1-3]: ", 1, 3)


def elegir_esquema():
    print("\nEsquema de derivacion (1ra derivada):")
    print("  1. Hacia adelante")
    print("  2. Central")
    print("  3. Hacia atras")
    opcion = leer_opcion("Elige [1-3]: ", 1, 3)
    return {1: "adelante", 2: "central", 3: "atras"}[opcion]


def resolver_constancia_h(h, h_detectado_constante, modo):
    """Define si se trabajara como h constante o no, segun menu."""
    h_promedio = float(np.mean(h))
    advertencia = None

    if modo == 1:
        usar_h_constante = h_detectado_constante
        origen = "automatico"
    elif modo == 2:
        usar_h_constante = True
        origen = "forzado"
        if not h_detectado_constante:
            advertencia = (
                "Se forzo h constante pero los datos no son uniformes. "
                f"Se usara h promedio = {h_promedio:.10g}."
            )
    else:
        usar_h_constante = False
        origen = "forzado"
        if h_detectado_constante:
            advertencia = (
                "Se forzo h no constante aunque los datos son uniformes. "
                "Se aplicaran formulas generales no uniformes."
            )

    return usar_h_constante, origen, h_promedio, advertencia


def polinomio_lagrange(x, y):
    """Construye el polinomio interpolante de Lagrange como np.poly1d."""
    n = x.size
    p = np.poly1d([0.0])

    for i in range(n):
        base = np.poly1d([1.0])
        denom = 1.0
        for j in range(n):
            if i == j:
                continue
            base *= np.poly1d([1.0, -x[j]])
            denom *= (x[i] - x[j])
        p += y[i] * (base / denom)

    return p


def formatear_polinomio(p, variable="x", digitos=10):
    """Convierte np.poly1d a un string legible."""
    coefs = np.asarray(p.c, dtype=float)
    grado = len(coefs) - 1
    terminos = []

    for k, coef in enumerate(coefs):
        exp = grado - k
        if abs(coef) < TOL:
            continue

        signo = "-" if coef < 0 else "+"
        mag = abs(coef)
        mag_str = f"{mag:.{digitos}g}"

        if exp == 0:
            termino = mag_str
        elif exp == 1:
            termino = variable if np.isclose(mag, 1.0, atol=TOL, rtol=0.0) else f"{mag_str}*{variable}"
        else:
            termino = (
                f"{variable}^{exp}"
                if np.isclose(mag, 1.0, atol=TOL, rtol=0.0)
                else f"{mag_str}*{variable}^{exp}"
            )

        terminos.append((signo, termino))

    if not terminos:
        return "0"

    partes = []
    for i, (signo, termino) in enumerate(terminos):
        if i == 0:
            partes.append(termino if signo == "+" else f"-{termino}")
        else:
            partes.append(f" {signo} {termino}")
    return "".join(partes)


def limpiar_ceros(valores, tol=TOL):
    arr = np.asarray(valores, dtype=float).copy()
    arr[np.abs(arr) < tol] = 0.0
    return arr


def derivadas_h_constante(x, y, esquema, h):
    """Derivadas de 1er orden para h constante con formulas clasicas de 3 puntos."""
    n = x.size
    resultados = []

    if esquema == "adelante":
        # f'(x_i) = (-3f_i + 4f_{i+1} - f_{i+2}) / (2h)
        for i in range(0, n - 2):
            d = (-3.0 * y[i] + 4.0 * y[i + 1] - y[i + 2]) / (2.0 * h)
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h,
                "h2": h,
                "nodos": f"x{i},x{i+1},x{i+2}",
            })

    elif esquema == "central":
        # f'(x_i) = (f_{i+1} - f_{i-1}) / (2h)
        for i in range(1, n - 1):
            d = (y[i + 1] - y[i - 1]) / (2.0 * h)
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h,
                "h2": h,
                "nodos": f"x{i-1},x{i},x{i+1}",
            })

    else:  # atras
        # f'(x_i) = (3f_i - 4f_{i-1} + f_{i-2}) / (2h)
        for i in range(2, n):
            d = (3.0 * y[i] - 4.0 * y[i - 1] + y[i - 2]) / (2.0 * h)
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h,
                "h2": h,
                "nodos": f"x{i-2},x{i-1},x{i}",
            })

    return resultados


def derivadas_h_no_constante(x, y, esquema):
    """Derivadas de 1er orden para h no constante usando 3 puntos por Lagrange local."""
    n = x.size
    resultados = []

    if esquema == "adelante":
        # x_i, x_{i+1}, x_{i+2}
        for i in range(0, n - 2):
            h1 = x[i + 1] - x[i]
            h2 = x[i + 2] - x[i + 1]
            c0 = -(2.0 * h1 + h2) / (h1 * (h1 + h2))
            c1 = (h1 + h2) / (h1 * h2)
            c2 = -h1 / (h2 * (h1 + h2))
            d = c0 * y[i] + c1 * y[i + 1] + c2 * y[i + 2]
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h1,
                "h2": h2,
                "nodos": f"x{i},x{i+1},x{i+2}",
            })

    elif esquema == "central":
        # x_{i-1}, x_i, x_{i+1}
        for i in range(1, n - 1):
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            c_izq = -h2 / (h1 * (h1 + h2))
            c0 = (h2 - h1) / (h1 * h2)
            c_der = h1 / (h2 * (h1 + h2))
            d = c_izq * y[i - 1] + c0 * y[i] + c_der * y[i + 1]
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h1,
                "h2": h2,
                "nodos": f"x{i-1},x{i},x{i+1}",
            })

    else:  # atras
        # x_{i-2}, x_{i-1}, x_i
        for i in range(2, n):
            h1 = x[i] - x[i - 1]
            h2 = x[i - 1] - x[i - 2]
            c0 = (2.0 * h1 + h2) / (h1 * (h1 + h2))
            c1 = -(h1 + h2) / (h1 * h2)
            c2 = h1 / (h2 * (h1 + h2))
            d = c0 * y[i] + c1 * y[i - 1] + c2 * y[i - 2]
            resultados.append({
                "i": i,
                "x": x[i],
                "d": d,
                "h1": h1,
                "h2": h2,
                "nodos": f"x{i-2},x{i-1},x{i}",
            })

    return resultados


def imprimir_tabla_puntos(x, y):
    print("\nPuntos ordenados:")
    print(f"  {'i':>4} {'x_i':>16} {'y_i':>16}")
    print("  " + "-" * 40)
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g}")


def imprimir_tabla_h(x, h):
    print("\nTabla de pasos h_i (pares consecutivos):")
    print(f"  {'i':>4} {'x_i':>16} {'x_(i+1)':>16} {'h_i':>16} {'h_i-h_0':>16}")
    print("  " + "-" * 76)
    h0 = h[0]
    for i, hi in enumerate(h):
        delta = hi - h0
        print(f"  {i:>4} {x[i]:>16.10g} {x[i+1]:>16.10g} {hi:>16.10g} {delta:>16.10g}")


def imprimir_tabla_resultados(resultados, dp_lagrange):
    print("\nResultados de la derivada (1ra):")
    encabezado = (
        f"  {'i':>4} {'x_i':>16} {'yprima aprox':>18} {'yprima Lag':>18} "
        f"{'Error abs':>14} {'h1':>14} {'h2':>14} {'Nodos':>16}"
    )
    print(encabezado)
    print("  " + "-" * 124)

    for r in resultados:
        xi = r["x"]
        d_aprox = float(r["d"])
        d_lag = float(dp_lagrange(xi))
        err = abs(d_aprox - d_lag)
        if abs(d_aprox) < TOL:
            d_aprox = 0.0
        if abs(d_lag) < TOL:
            d_lag = 0.0
        if abs(err) < TOL:
            err = 0.0

        print(
            f"  {r['i']:>4} {xi:>16.10g} {d_aprox:>18.10g} {d_lag:>18.10g} {err:>14.6e} {r['h1']:>14.10g} {r['h2']:>14.10g} {r['nodos']:>16}"
        )


def modulo_derivacion_desde_puntos():
    print("\n=== MODULO 1: DERIVACION NUMERICA DESDE PUNTOS ===\n")

    try:
        x = leer_arreglo("x")
        y = leer_arreglo("y")
        x, y = validar_y_ordenar_puntos(x, y)
    except ValueError as e:
        print(e)
        return

    imprimir_tabla_puntos(x, y)

    h, h0, detectado_constante, error_max = calcular_h(x)
    imprimir_tabla_h(x, h)

    print("\nDeteccion automatica de h:")
    if detectado_constante:
        print(f"  h constante detectado (h = {h0:.10g}).")
    else:
        print("  h no constante detectado.")
        print(f"  h_0 = {h0:.10g}, desviacion maxima = {error_max:.6e}")

    modo_h = elegir_modo_h()
    usar_h_constante, origen, h_promedio, advertencia = resolver_constancia_h(
        h, detectado_constante, modo_h
    )
    esquema = elegir_esquema()

    print("\nConfiguracion elegida:")
    tipo_h = "constante" if usar_h_constante else "no constante"
    print(f"  Tipo de h: {tipo_h} ({origen})")
    print(f"  Esquema: {esquema}")
    if advertencia is not None:
        print(f"  Aviso: {advertencia}")

    p_lagrange = polinomio_lagrange(x, y)
    dp_lagrange = np.polyder(p_lagrange)
    print("\nPolinomio interpolante de Lagrange global:")
    print(f"  P_{x.size - 1}(x) = {formatear_polinomio(p_lagrange)}")
    print("Derivada del polinomio global:")
    print(f"  P_{x.size - 1}'(x) = {formatear_polinomio(dp_lagrange)}")

    if usar_h_constante:
        resultados = derivadas_h_constante(x, y, esquema, h_promedio)
    else:
        resultados = derivadas_h_no_constante(x, y, esquema)

    if not resultados:
        print("\nNo hay suficientes nodos para el esquema elegido.")
        return

    imprimir_tabla_resultados(resultados, dp_lagrange)


def leer_coeficientes_polinomio():
    print("\nIngresa coeficientes de P(x) en orden descendente.")
    print("Ejemplo: 1 -3 2  representa  x^2 - 3x + 2")
    texto = input("coeficientes = ").strip()

    try:
        coefs = np.array([float(v) for v in texto.split()], dtype=float)
    except ValueError as exc:
        raise ValueError("Error: coeficientes invalidos.") from exc

    if coefs.size == 0:
        raise ValueError("Error: debes ingresar al menos un coeficiente.")

    return coefs


def modulo_polinomio():
    print("\n=== MODULO 2: DERIVACION DESDE POLINOMIO INGRESADO ===")

    try:
        coefs = leer_coeficientes_polinomio()
    except ValueError as e:
        print(e)
        return

    p = np.poly1d(coefs)
    dp = np.polyder(p)

    print("\nPolinomio ingresado:")
    print(f"  P(x)  = {formatear_polinomio(p)}")
    print("Derivada del polinomio:")
    print(f"  P'(x) = {formatear_polinomio(dp)}")

    try:
        x_eval = leer_arreglo("Puntos x para evaluar P(x) y P'(x)")
    except ValueError as e:
        print(e)
        return

    y_eval = limpiar_ceros(p(x_eval))
    dy_eval = limpiar_ceros(dp(x_eval))

    print("\nTabla de evaluacion del polinomio:")
    print(f"  {'i':>4} {'x_i':>16} {'P(x_i)':>16} {'P\'(x_i)':>16}")
    print("  " + "-" * 60)
    for i, (xi, yi, dyi) in enumerate(zip(x_eval, y_eval, dy_eval)):
        print(f"  {i:>4} {xi:>16.10g} {yi:>16.10g} {dyi:>16.10g}")

    # Reconstruccion del polinomio desde los resultados (x_i, P(x_i)).
    xu, indices = np.unique(x_eval, return_index=True)
    yu = y_eval[indices]

    if xu.size >= 2:
        p_rec = polinomio_lagrange(xu, yu)
        dp_rec = np.polyder(p_rec)
        print("\nPolinomio recuperado desde los resultados (Lagrange):")
        print(f"  P_rec(x)  = {formatear_polinomio(p_rec)}")
        print(f"  P_rec'(x) = {formatear_polinomio(dp_rec)}")
    else:
        print("\nNo se puede reconstruir polinomio: se necesitan al menos 2 x distintos.")


def main():
    print("\n=== DERIVACION NUMERICA - MENU PRINCIPAL ===\n")
    print("  1. Derivacion numerica desde puntos (h cte/no cte)")
    print("  2. Modulo polinomio (ingresar P(x), evaluar y derivar)")

    opcion = leer_opcion("Elige [1-2]: ", 1, 2)

    if opcion == 1:
        modulo_derivacion_desde_puntos()
    else:
        modulo_polinomio()


if __name__ == "__main__":
    main()
