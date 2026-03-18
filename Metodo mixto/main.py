import math
import re


# ─────────────────────────────────────────
# Utilidades comunes
# ─────────────────────────────────────────

def _crear_funcion_segura(f_str):
    # Reemplaza ^ por ** y ln( por math.log( para compatibilidad
    expr = f_str.strip().replace("^", "**").replace("ln(", "math.log(")

    # Entorno de evaluacion controlado: solo se permiten estas funciones/constantes
    allowed = {
        "__builtins__": {}, "math": math, "abs": abs, "pow": pow,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "pi": math.pi, "e": math.e,
    }
    return lambda x: eval(expr, allowed, {"x": x})


def _derivadas_simbolicas(expr_str):
    # Usa sympy para calcular f'(x) y f''(x) de forma simbolica
    try:
        from sympy import symbols, diff, lambdify
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
    except Exception as e:
        raise RuntimeError("Falta sympy. Instalalo con: pip install sympy") from e

    x = symbols("x")

    # Adapta la expresion al formato que entiende sympy
    s = expr_str.strip().replace("^", "**").replace("ln(", "log(")
    s = s.replace("math.pi", "pi")
    s = re.sub(r"\bmath\.e\b", "E", s)
    s = re.sub(r"\bmath\.", "", s)

    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(s, transformations=transformations)

    # Calcula primera y segunda derivada simbolica
    df_expr  = diff(expr, x)
    d2f_expr = diff(df_expr, x)

    # Convierte a funciones numericas usando math
    df_func  = lambdify(x, df_expr,  modules="math")
    d2f_func = lambdify(x, d2f_expr, modules="math")
    return df_func, str(df_expr), d2f_func, str(d2f_expr)


def _actualizar_bracket(f, state, x_new):
    # Intenta estrechar el intervalo [a,b] usando el nuevo punto
    # Si f(x_new) tiene signo opuesto a f(a), el nuevo b es x_new, y viceversa
    try:
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            return
        if state["fa"] * fx_new < 0:
            state["b"]  = x_new
            state["fb"] = fx_new
        elif state["fb"] * fx_new < 0:
            state["a"]  = x_new
            state["fa"] = fx_new
        # Si no hay cambio de signo con ninguno, el intervalo no se actualiza
    except Exception:
        pass


def _en(x_nuevo, x_actual):
    # Calcula el error relativo: |x_nuevo - x_actual| / |x_nuevo|
    # Si x_nuevo es muy cercano a cero, usa error absoluto para evitar division por cero
    if abs(x_nuevo) > 1e-15:
        return abs(x_nuevo - x_actual) / abs(x_nuevo)
    return abs(x_nuevo - x_actual)


# ─────────────────────────────────────────
# Pasos individuales (una iteracion cada uno)
# Cada funcion ejecuta UN paso del metodo correspondiente
# y devuelve (x_nuevo, f(x_nuevo))
# ─────────────────────────────────────────

def paso_biseccion(f, state):
    a, b, fa, fb = state["a"], state["b"], state["fa"], state["fb"]

    # Verifica que el intervalo siga teniendo cambio de signo
    if fa * fb > 0:
        raise ValueError("El intervalo [a,b] ya no contiene un cambio de signo.")

    # Punto medio del intervalo actual
    c  = (a + b) / 2
    fc = f(c)

    # Actualiza el intervalo: conserva el lado donde hay cambio de signo
    if fa * fc < 0:
        state["b"] = c;  state["fb"] = fc   # raiz en [a, c]
    else:
        state["a"] = c;  state["fa"] = fc   # raiz en [c, b]

    return c, fc


def paso_regla_falsa(f, state):
    a, b, fa, fb = state["a"], state["b"], state["fa"], state["fb"]

    # Verifica que el intervalo siga teniendo cambio de signo
    if fa * fb > 0:
        raise ValueError("El intervalo [a,b] ya no contiene un cambio de signo.")

    denom = fb - fa
    if abs(denom) < 1e-15:
        raise ValueError("f(b) - f(a) ~ 0 en regla falsa.")

    # Interseccion de la secante entre (a,f(a)) y (b,f(b)) con el eje x
    c  = b - fb * (b - a) / denom
    fc = f(c)

    # Actualiza el intervalo igual que en biseccion
    if fa * fc < 0:
        state["b"] = c;  state["fb"] = fc
    else:
        state["a"] = c;  state["fa"] = fc

    return c, fc


def paso_newton_raphson(f, df, state):
    x   = state["x"]
    fx  = f(x)
    dfx = df(x)

    if not (math.isfinite(fx) and math.isfinite(dfx)):
        raise ValueError("f(x) o f'(x) no es finito.")
    if abs(dfx) < 1e-15:
        raise ValueError("f'(x) ~ 0, derivada nula.")

    # x_{n+1} = x_n - f(x_n) / f'(x_n)
    x_new  = x - fx / dfx
    fx_new = f(x_new)

    # Intenta actualizar el intervalo con el nuevo punto
    _actualizar_bracket(f, state, x_new)
    return x_new, fx_new


def paso_newton_mejorado(f, df, d2f, state):
    x    = state["x"]
    fx   = f(x)
    dfx  = df(x)
    d2fx = d2f(x)

    if not (math.isfinite(fx) and math.isfinite(dfx) and math.isfinite(d2fx)):
        raise ValueError("f(x), f'(x) o f''(x) no es finito.")

    # Denominador: (f')^2 - f*f''
    denom = dfx * dfx - fx * d2fx
    if abs(denom) < 1e-15:
        raise ValueError("Denominador ~ 0 en Newton mejorado.")

    # x_{n+1} = x_n - f*f' / ((f')^2 - f*f'')
    x_new  = x - (fx * dfx) / denom
    fx_new = f(x_new)

    # Intenta actualizar el intervalo con el nuevo punto
    _actualizar_bracket(f, state, x_new)
    return x_new, fx_new


def paso_secante(f, state):
    x_act = state["x"]       # punto actual x_n
    x_ant = state["x_prev"]  # punto anterior x_{n-1}
    f_act = f(x_act)
    f_ant = f(x_ant)

    if not (math.isfinite(f_act) and math.isfinite(f_ant)):
        raise ValueError("f(x) no es finito en secante.")

    # Denominador: f(x_n) - f(x_{n-1})
    denom = f_act - f_ant
    if abs(denom) < 1e-15:
        raise ValueError("f(x_n) - f(x_{n-1}) ~ 0 en secante.")

    # x_{n+1} = x_n - f(x_n)*(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    x_new  = x_act - f_act * (x_act - x_ant) / denom
    fx_new = f(x_new)

    # Intenta actualizar el intervalo con el nuevo punto
    _actualizar_bracket(f, state, x_new)
    return x_new, fx_new


def paso_punto_fijo(g, state):
    x     = state["x"]

    # x_{n+1} = g(x_n)
    x_new = g(x)
    if not math.isfinite(x_new):
        raise ValueError("g(x) no es finito. El metodo diverge.")

    # Devuelve el nuevo punto (f(x_new) se evalua en el motor principal)
    return x_new, None


# ─────────────────────────────────────────
# Motor del metodo mixto
# ─────────────────────────────────────────

# Catalogo de metodos disponibles
MENU = {
    "1": "Biseccion",
    "2": "Regla Falsa",
    "3": "Newton-Raphson",
    "4": "Newton Mejorado",
    "5": "Secante",
    "6": "Punto Fijo",
}

# Conjuntos que indican que recursos necesita cada metodo
NECESITA_INTERVALO = {"1", "2"}   # requieren [a,b] con cambio de signo
NECESITA_DERIVADA  = {"3", "4"}   # requieren f'(x), f''(x) via sympy
NECESITA_D2        = {"4"}        # requiere ademas f''(x)
NECESITA_SECANTE   = {"5"}        # requiere dos puntos iniciales
NECESITA_G         = {"6"}        # requiere g(x) en lugar de f(x)


def metodo_mixto(f, df, d2f, g, secuencia, state, tolerancia, max_iteraciones):
    n = len(secuencia)  # numero de metodos en la secuencia (se cicla)

    # Cabecera de la tabla de iteraciones
    print(f"\n{'Iter':<6} {'Metodo':<18} {'x_n':<22} {'x_{n+1}':<22} {'f(x_{n+1})':<20} {'EN':<15}")
    print("-" * 108)

    for iteracion in range(max_iteraciones):
        # Selecciona el metodo de esta iteracion (ciclo sobre la secuencia)
        clave   = secuencia[iteracion % n]
        nombre  = MENU[clave]
        x_antes = state["x"]  # guarda x_n antes del paso

        # Ejecuta el paso del metodo correspondiente
        try:
            if clave == "1":
                x_nuevo, fx_nuevo = paso_biseccion(f, state)
            elif clave == "2":
                x_nuevo, fx_nuevo = paso_regla_falsa(f, state)
            elif clave == "3":
                x_nuevo, fx_nuevo = paso_newton_raphson(f, df, state)
            elif clave == "4":
                x_nuevo, fx_nuevo = paso_newton_mejorado(f, df, d2f, state)
            elif clave == "5":
                x_nuevo, fx_nuevo = paso_secante(f, state)
            elif clave == "6":
                # Punto fijo solo devuelve x_nuevo; evaluamos f(x_nuevo) aqui
                x_nuevo, _ = paso_punto_fijo(g, state)
                fx_nuevo = f(x_nuevo)
        except Exception as e:
            print(f"\nError en iteracion {iteracion} ({nombre}): {e}")
            return None

        # Calcula el error relativo respecto al punto anterior
        EN = _en(x_nuevo, x_antes)

        # Imprime la fila de esta iteracion (sin EN en la primera)
        if iteracion == 0:
            print(f"{iteracion:<6} {nombre:<18} {x_antes:<22.10f} {x_nuevo:<22.10f} {fx_nuevo:<20.10e} {'---':<15}")
        else:
            print(f"{iteracion:<6} {nombre:<18} {x_antes:<22.10f} {x_nuevo:<22.10f} {fx_nuevo:<20.10e} {EN:<15.8e}")

        # Actualiza el estado compartido para el siguiente metodo
        state["x_prev"] = state["x"]   # x_n pasa a ser x_{n-1} (necesario para secante)
        state["x"]      = x_nuevo      # x_{n+1} pasa a ser el nuevo x_n

        # Criterio de parada: el error relativo ya es menor que la tolerancia
        if iteracion > 0 and EN < tolerancia:
            print(f"\nRaiz encontrada: x = {x_nuevo:.10f}")
            print(f"f({x_nuevo:.10f}) = {f(x_nuevo):.10e}")
            print(f"EN = {EN:.10e}")
            print(f"Iteraciones: {iteracion + 1}")
            return x_nuevo

    # Si se agotaron las iteraciones, reporta la mejor aproximacion
    print(f"\nAdvertencia: Se alcanzo el numero maximo de iteraciones ({max_iteraciones})")
    x_final = state["x"]
    print(f"Raiz aproximada: x = {x_final:.10f}")
    print(f"f({x_final:.10f}) = {f(x_final):.10e}")
    return x_final


# ─────────────────────────────────────────
# Entrada de datos
# ─────────────────────────────────────────

def seleccionar_metodos():
    print("\nMetodos disponibles:")
    for k, v in MENU.items():
        print(f"  {k}. {v}")
    print()
    print("Escribe los numeros de los metodos en el orden que quieras alternar.")
    print("Ejemplo: '1 3' alterna Biseccion -> Newton-Raphson -> Biseccion -> ...")
    raw = input("Secuencia: ").strip().split()

    # Filtra solo los numeros validos del menu
    secuencia = [r for r in raw if r in MENU]
    if not secuencia:
        print("No se reconocio ninguna opcion valida.")
        return None
    print("\nSecuencia seleccionada: " + " -> ".join(MENU[k] for k in secuencia) + " -> (ciclo)")
    return secuencia


def main():
    print("\n=== METODO MIXTO ===")
    print("Combina varios metodos de busqueda de raices alternandolos cada iteracion.\n")

    # 1. Funcion f(x): base para todos los metodos excepto punto fijo
    f_str = input("f(x) = ").strip()
    try:
        f = _crear_funcion_segura(f_str)
        f(1)
    except Exception as e:
        print(f"Error al interpretar f(x): {e}")
        return

    # 2. El usuario elige los metodos y su orden
    secuencia = seleccionar_metodos()
    if secuencia is None:
        return

    claves = set(secuencia)  # conjunto de claves unicas para saber que se necesita

    # 3. Calcula derivadas solo si alguno de los metodos seleccionados las necesita
    df = d2f = None
    if claves & NECESITA_DERIVADA:
        try:
            df, df_str, d2f, d2f_str = _derivadas_simbolicas(f_str)
            df(1); d2f(1)
            print(f"\nf'(x)  = {df_str}")
            print(f"f''(x) = {d2f_str}")
        except Exception as e:
            print(f"Error al calcular derivadas: {e}")
            return

    # 4. Pide g(x) solo si punto fijo esta en la secuencia
    g = None
    if claves & NECESITA_G:
        print("\nPunto fijo requiere g(x) tal que x = g(x).")
        g_str = input("g(x) = ").strip()
        try:
            g = _crear_funcion_segura(g_str)
            g(1)
        except Exception as e:
            print(f"Error al interpretar g(x): {e}")
            return

    # 5. Intervalo [a, b]: necesario para biseccion y regla falsa
    #    Tambien sirve como estado inicial para todos los metodos abiertos
    print("\nIngresa el intervalo inicial [a, b]:")
    try:
        a = float(input("a = "))
        b = float(input("b = "))
        if a >= b:
            print("Error: 'a' debe ser menor que 'b'")
            return
    except ValueError:
        print("Error: ingresa numeros validos")
        return

    fa = f(a); fb = f(b)

    # Si se usan metodos de intervalo, verifica la condicion de Bolzano
    if (claves & NECESITA_INTERVALO) and fa * fb > 0:
        print(f"Error: f(a) y f(b) deben tener signos opuestos para los metodos de intervalo.")
        print(f"f({a}) = {fa},  f({b}) = {fb}")
        return

    # 6. Punto inicial x0: valor de arranque para los metodos abiertos
    x0_str = input(f"\nPunto inicial x0 (Enter para usar el punto medio {(a+b)/2:.6f}): ").strip()
    try:
        x0 = float(x0_str) if x0_str else (a + b) / 2
    except ValueError:
        x0 = (a + b) / 2

    # 7. Tolerancia: criterio de parada por error relativo
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

    # 8. Estado inicial compartido entre todos los metodos
    state = {
        "x":      x0,   # mejor aproximacion actual
        "x_prev": a,    # punto anterior (x_{n-1}), usado por secante en la primera iteracion
        "a":      a,    # extremo izquierdo del intervalo actual
        "b":      b,    # extremo derecho del intervalo actual
        "fa":     fa,   # f(a)
        "fb":     fb,   # f(b)
    }

    secuencia_nombres = " -> ".join(MENU[k] for k in secuencia)
    print(f"\nResolviendo: f(x) = {f_str}")
    print(f"Secuencia:   {secuencia_nombres} -> (ciclo)")
    print(f"x0 = {x0},  intervalo = [{a}, {b}]")

    metodo_mixto(f, df, d2f, g, secuencia, state, tolerancia, max_iteraciones)


if __name__ == "__main__":
    main()
