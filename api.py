"""api.py — Capa publica explicita de los modulos numericos.

main.py usa este modulo para mantener compatibilidad con codigo y tests que
acceden a funciones como main._biseccion_core o main.metodo_rk4.
"""

import common as _common
import tema_derivacion_integracion as _di
import tema_edo as _edo
import tema_edp as _edp
import tema_interpolacion as _interp
import tema_raices as _raices

_EXPORTS = {
    _common: [
        "_MPLOK", "_SAFE_XY", "_en", "_pedir_funcion", "_pedir_intervalo",
        "_pedir_tol_iter", "_preguntar_grafica", "AYUDA_FUNCIONES",
        "buscar_cambios_de_signo", "crear_evaluador_seguro",
        "crear_funcion_segura", "crear_funcion_xy_segura",
        "derivada_simbolica", "derivadas_simbolicas", "math", "np",
        "refinar_cambio", "sugerir_intervalos",
    ],
    _raices: [
        "_MM_G", "_MM_DERIVADA", "_MM_INTERVALO", "_MM_MENU",
        "_biseccion_core", "_contar_cambios_signo", "_division_sintetica",
        "_divisores", "_horner_poly", "_mixto_core", "_mm_actualizar",
        "_mm_biseccion", "_mm_newton", "_mm_newton2", "_mm_punto_fijo",
        "_mm_regla_falsa", "_mm_secante", "_newton_core",
        "_newton_mejorado_core", "_poly_str", "_punto_fijo_core",
        "_regla_falsa_core", "_secante_core", "menu_biseccion",
        "menu_bolzano", "menu_graficar_raices", "menu_mixto",
        "menu_newton_mejorado", "menu_newton_raphson",
        "menu_polinomios_enteros", "menu_punto_fijo", "menu_raices",
        "menu_regla_falsa", "menu_secante",
    ],
    _interp: [
        "_eval_poli", "_ingresar_datos_xy", "_print_resultado",
        "_print_lagrange_bases", "_print_tabla_dif", "_print_tabla_xy",
        "_print_tabla_reg_exponencial", "_print_tabla_reg_lineal",
        "_print_tabla_reg_multiple", "_print_tabla_reg_polinomial",
        "_tabla_dif_div",
        "interpolacion_lagrange", "interpolacion_newton",
        "menu_graficar_interp", "menu_interp_aprox",
        "regresion_exponencial", "regresion_funcion_conocida",
        "regresion_lineal", "regresion_multiple", "regresion_polinomial",
    ],
    _di: [
        "_ESQUEMAS_D1_PT", "_ESQUEMAS_D2_PT", "_INTEG_COMP",
        "_INTEG_OPS", "_TOL_NUM", "_abiertas_tabular",
        "_aproximar_derivada_punto", "_aproximar_derivada_punto_tabla",
        "_calc_h", "_d1_hcte", "_d1_hvar", "_d2_hcte", "_d2_hvar",
        "_d3_hcte", "_d4_hcte", "_fmt_poly", "_integ_dos_puntos",
        "_integ_punto_medio", "_integ_simpson13_comp_f",
        "_integ_simpson13_simple", "_integ_simpson38_comp_f",
        "_integ_simpson38_simple", "_integ_trapecio_comp_f",
        "_integ_trapecio_simple", "_interpolar_uniforme", "_leer_arr",
        "_modulo_derivacion_puntos", "_modulo_integracion",
        "_modulo_polinomio", "_nodos_vals", "_pedir_esquema",
        "_pedir_modo_h", "_polinomio_lag", "_print_d1", "_print_d2",
        "_print_d3", "_print_d4", "_print_nodos", "_print_nodos_pesos", "_print_tabla_h",
        "_resolver_modo_h", "_simpson13", "_simpson38", "_trapecio",
        "_validar_ord", "calcular_integracion_funcion",
        "menu_derivacion_integracion", "menu_graficar_integracion",
    ],
    _edo: [
        "_AYUDA_TXY", "_AYUDA_TXV", "_AYUDA_XY", "_ResultadoEDO",
        "_crear_func_txy", "_crear_func_txv", "_e2_euler_paso",
        "_e2_rk2pm_paso", "_e2_rk4_paso", "_edo2_run", "_edo_header",
        "_pedir_ci", "_pedir_ci_edo2", "_pedir_ci_sistema",
        "_pedir_edo2", "_pedir_ejes_2orden", "_pedir_ejes_sistema",
        "_pedir_ode", "_pedir_sistema2", "_s2_euler_paso",
        "_s2_heun_paso", "_s2_rk2pm_paso", "_s2_rk4_paso",
        "_s2_paso_detalle", "_sistema2_run", "_validar_paso", "comparacion_metodos_edo",
        "menu_graficar_odes", "menu_odes", "metodo_euler",
        "metodo_rk2_heun", "metodo_rk2_pm", "metodo_rk4",
    ],
    _edp: [
        "_edp_comparar_gs_sor", "_edp_datos", "_edp_imprimir",
        "_edp_print_convergencia", "_edp_run", "_edp_solver",
        "_pedir_bc", "menu_edp", "menu_graficar_edp",
    ],
}

__all__ = []

for _module, _names in _EXPORTS.items():
    for _name in _names:
        globals()[_name] = getattr(_module, _name)
        __all__.append(_name)

del _module, _name, _names
