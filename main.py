"""main.py — Menu principal de metodos numericos.

Los metodos estan separados por tema en archivos independientes:
- tema_raices.py
- tema_interpolacion.py
- tema_derivacion_integracion.py
- tema_edo.py
- tema_edp.py
"""

import api as _api
from common import VolverAtras, _input
from tema_derivacion_integracion import menu_derivacion_integracion, menu_graficar_integracion
from tema_edo import menu_graficar_odes, menu_odes
from tema_edp import menu_edp, menu_graficar_edp
from tema_interpolacion import menu_graficar_interp, menu_interp_aprox
from tema_raices import menu_graficar_raices, menu_raices


def __getattr__(name):
    """Mantiene compatibilidad con codigo que usa funciones como main._biseccion_core."""
    if name in _api.__all__:
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def menu_graficar_global():
    while True:
        print("\n=== GRAFICAR / COMPARAR METODOS ===\n")
        print("  1. Raices de ecuaciones")
        print("  2. Interpolacion y regresion")
        print("  3. Integracion")
        print("  4. EDOs")
        print("  5. EDPs")
        print("  0. Volver al menu principal")
        try:
            op = _input("\nElige [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError, VolverAtras):
            break
        if op == "0":
            break
        elif op == "1":
            menu_graficar_raices()
        elif op == "2":
            menu_graficar_interp()
        elif op == "3":
            menu_graficar_integracion()
        elif op == "4":
            menu_graficar_odes()
        elif op == "5":
            menu_graficar_edp()
        else:
            print("Opcion invalida. Elige entre 0 y 5.")


def main():
    while True:
        print("\n" + "=" * 60)
        print("  METODOS NUMERICOS — MENU PRINCIPAL")
        print("=" * 60)
        print("  1. Raices de ecuaciones")
        print("  2. Interpolacion y Aproximacion   (Tema 3)")
        print("  3. Derivacion e Integracion       (Tema 4)")
        print("  4. Ecuaciones Diferenciales       (Tema 5)")
        print("  5. EDPs Laplace / Poisson         (Tema 6)")
        print("  6. Graficar / Comparar metodos")
        print("  0. Salir")
        print("=" * 60)
        try:
            op = _input("Elige [0-6]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego."); break
        except VolverAtras:
            continue
        if op == "0":
            print("Hasta luego."); break
        elif op == "1":
            try: menu_raices()
            except VolverAtras: pass
        elif op == "2":
            try: menu_interp_aprox()
            except VolverAtras: pass
        elif op == "3":
            try: menu_derivacion_integracion()
            except VolverAtras: pass
        elif op == "4":
            try: menu_odes()
            except VolverAtras: pass
        elif op == "5":
            try: menu_edp()
            except VolverAtras: pass
        elif op == "6":
            try: menu_graficar_global()
            except VolverAtras: pass
        else:
            print("Opcion invalida. Elige entre 0 y 6.")




if __name__ == "__main__":
    main()
