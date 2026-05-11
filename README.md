# Programas de aproximacion numerica

Coleccion interactiva de metodos numericos organizada por temas. El punto de
entrada sigue siendo `main.py`, pero la implementacion esta separada en modulos
para facilitar lectura, pruebas y mantenimiento.

## Instalacion

Se recomienda usar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

```bash
python3 main.py
```

Durante la entrada interactiva, si un dato se escribe mal el programa vuelve a
pedir ese mismo dato. Escribe `esc` para volver atras desde un prompt.

El menu principal contiene:

- Raices de ecuaciones.
- Interpolacion y aproximacion.
- Derivacion e integracion.
- Ecuaciones diferenciales ordinarias.
- EDPs de Laplace y Poisson.
- Graficar / comparar metodos.

## Estructura

- `main.py`: menu principal.
- `api.py`: capa de compatibilidad para exponer funciones desde `main`.
- `common.py`: utilidades compartidas, entrada comun y graficacion opcional.
- `tema_raices.py`: Bolzano, biseccion, regla falsa, punto fijo, secante, Newton y polinomios.
- `tema_interpolacion.py`: regresiones e interpolacion de Newton/Lagrange.
- `tema_derivacion_integracion.py`: derivacion numerica e integracion Newton-Cotes.
- `tema_edo.py`: Euler, RK2, Heun, RK4, sistemas 2D y EDOs de segundo orden.
- `tema_edp.py`: Laplace/Poisson por diferencias finitas.
- `graficar.py`: funciones de visualizacion con `matplotlib`.
- `utils.py`: evaluador matematico seguro y utilidades simbolicas.

## Expresiones matematicas

Las funciones aceptan expresiones como:

```text
x^3 - 2*x - 5
sin(x) - x/2
exp(x) - 3*x
ln(x)
math.sin(pi*x)
```

El evaluador permite solo numeros, variables esperadas, operadores aritmeticos y
funciones matematicas de lista blanca. Llamadas como `__import__(...)` quedan
bloqueadas.

## Graficas

Puedes graficar despues de resolver un metodo cuando el programa pregunte
`Graficar resultado?`, o entrar directamente en:

```text
6. Graficar / Comparar metodos
```

Desde ahi puedes comparar raices, interpolacion/regresion, integracion, EDOs y
EDPs segun el tipo de problema.

## Salidas utiles para examen

Ademas del resultado final, varios metodos imprimen tablas intermedias que suelen
pedirse en ejercicios:

- Regresion lineal, polinomial, exponencial y multiple: tabla auxiliar de sumas
  y sistema normal.
- Interpolacion de Newton: opcion para mostrar la tabla de diferencias divididas.
- Interpolacion de Lagrange: opcion para mostrar los polinomios base `L_i(x)`.
- Integracion compuesta: nodos, valores `f(xi)`, pesos y suma ponderada.
- Sistemas de EDOs: detalle de `k1`, `k2`, `k3`, `k4` segun el metodo.
- EDPs: tabla de convergencia por iteracion y comparacion Gauss-Seidel vs SOR.

## Tests

```bash
python3 test_debug.py
python3 test_edp.py
python3 test_menu.py
python3 test_stress_10.py
```

Tambien puedes verificar sintaxis/importaciones:

```bash
python3 -m py_compile main.py api.py common.py tema_*.py utils.py graficar.py
```
