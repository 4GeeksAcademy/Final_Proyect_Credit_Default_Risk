from __future__ import annotations

# Diccionario de mapeo de columnas para visualización en SHAP
COLUMN_MAPPING = {
    # ==================== DATOS BASE ====================
    'SK_ID_CURR': 'ID del cliente',
    'NAME_CONTRACT_TYPE': 'Tipo de crédito',
    'AMT_CREDIT': 'Dinero que pide',
    'AMT_ANNUITY': 'Cuota mensual',
    'AMT_INCOME_TOTAL': 'Dinero que gana al año',
    'AMT_GOODS_PRICE': 'Valor del bien financiado',
    'DAYS_BIRTH': 'Edad',
    'DAYS_EMPLOYED': 'Tiempo trabajando',
    'CODE_GENDER': 'Género',
    'FLAG_OWN_CAR': 'Tiene coche',
    'FLAG_OWN_REALTY': 'Tiene vivienda',
    'CNT_CHILDREN': 'Número de hijos',
    'CNT_FAM_MEMBERS': 'Personas en casa',
    'NAME_TYPE_SUITE': 'Viene con alguien',
    'NAME_INCOME_TYPE': 'De dónde sale su dinero',
    'NAME_EDUCATION_TYPE': 'Nivel de estudios',
    'NAME_FAMILY_STATUS': 'Situación familiar',
    'NAME_HOUSING_TYPE': 'Tipo de vivienda',
    'REGION_POPULATION_RELATIVE': 'Tamaño de la zona donde vive',
    'DAYS_REGISTRATION': 'Hace cuánto se registró',
    'DAYS_ID_PUBLISH': 'Antigüedad del documento',
    'OWN_CAR_AGE': 'Años del coche',
    'FLAG_MOBIL': 'Tiene móvil',
    'FLAG_EMP_PHONE': 'Teléfono del trabajo',
    'FLAG_WORK_PHONE': 'Teléfono laboral',
    'FLAG_CONT_MOBILE': 'Móvil de contacto',
    'FLAG_PHONE': 'Tiene teléfono fijo',
    'FLAG_EMAIL': 'Tiene email',
    'OCCUPATION_TYPE': 'Trabajo',
    'REGION_RATING_CLIENT': 'Cómo de buena es la zona',
    'REGION_RATING_CLIENT_W_CITY': 'Cómo de buena es la zona (con ciudad)',
    'WEEKDAY_APPR_PROCESS_START': 'Día que pidió el crédito',
    'HOUR_APPR_PROCESS_START': 'Hora que pidió el crédito',
    'REG_REGION_NOT_LIVE_REGION': 'Registrado en otra región',
    'REG_REGION_NOT_WORK_REGION': 'Trabaja en otra región',
    'LIVE_REGION_NOT_WORK_REGION': 'Vive y trabaja en regiones distintas',
    'REG_CITY_NOT_LIVE_CITY': 'Registrado en otra ciudad',
    'REG_CITY_NOT_WORK_CITY': 'Trabaja en otra ciudad',
    'LIVE_CITY_NOT_WORK_CITY': 'Vive y trabaja en ciudades distintas',
    'ORGANIZATION_TYPE': 'Tipo de empresa',
    'EXT_SOURCE_1': 'Puntuación externa 1',
    'EXT_SOURCE_2': 'Puntuación externa 2',
    'EXT_SOURCE_3': 'Puntuación externa 3',
    'APARTMENTS_AVG': 'Apartamentos del edificio',
    'BASEMENTAREA_AVG': 'Tamaño del sótano',
    'YEARS_BEGINEXPLUATATION_AVG': 'Antigüedad del edificio',
    'YEARS_BUILD_AVG': 'Año de construcción',
    'COMMONAREA_AVG': 'Zonas comunes',
    'ELEVATORS_AVG': 'Ascensores',
    'ENTRANCES_AVG': 'Entradas del edificio',
    'FLOORSMAX_AVG': 'Número máximo de pisos',
    'FLOORSMIN_AVG': 'Número mínimo de pisos',
    'LANDAREA_AVG': 'Tamaño del terreno',
    'LIVINGAPARTMENTS_AVG': 'Viviendas habitadas',
    'LIVINGAREA_AVG': 'Superficie habitable',
    'NONLIVINGAPARTMENTS_AVG': 'Viviendas no habitadas',
    'NONLIVINGAREA_AVG': 'Superficie no habitable',

    
    # ==================== EXT SOURCE - ESTADÍSTICAS ====================
    'EXT_SOURCE_MEAN': 'Nota de otros bancos (promedio)',
    'EXT_SOURCE_MIN': 'Nota más baja de otros bancos',
    'EXT_SOURCE_MAX': 'Nota más alta de otros bancos',
    'EXT_SOURCE_SUM': 'Suma de notas de otros bancos',
    'EXT_mean': 'Promedio de notas de otros bancos',
    'EXT_std': 'Variación entre notas de otros bancos',
    'EXT_min': 'Nota más baja de otros bancos',
    'EXT_max': 'Nota más alta de otros bancos',
    'EXT_sum': 'Suma total de notas de otros bancos',
    'EXT_prod': 'Notas de otros bancos combinadas',
    'EXT_range': 'Diferencia entre la nota más alta y la más baja',

    
    # ==================== EXT SOURCE - COMBINACIONES PONDERADAS ====================
    'EXT_w_01_06_03': 'Nota de otros bancos (peso principal en la 2ª)',
    'EXT_w_005_07_025': 'Nota de otros bancos (peso muy alto en la 2ª)',
    'EXT_w_015_055_030': 'Nota de otros bancos (peso equilibrado)',
    
    # ==================== EXT SOURCE - PRODUCTOS CRUZADOS ====================
    'EXT_1x2': 'Combinación de dos notas externas',
    'EXT_2x3': 'Combinación de dos notas externas (alternativa)',
    'EXT_1x3': 'Combinación de dos notas externas (otra fuente)',
    'EXT_1x2x3': 'Combinación de todas las notas externas',
    
    # ==================== EXT SOURCE - POTENCIAS Y TRANSFORMACIONES ====================
    'EXT_2_pow2': 'Nota de otros bancos (refuerza valores altos)',
    'EXT_2_pow3': 'Nota de otros bancos (refuerza mucho valores altos)',
    'EXT_2_pow05': 'Nota de otros bancos (suaviza valores)',
    'EXT_mean_pow2': 'Promedio de notas externas (refuerzo)',
    'EXT_mean_pow3': 'Promedio de notas externas (refuerzo fuerte)',
    'EXT_mean_pow05': 'Promedio de notas externas (suavizado)',
    'EXT_3_pow2': 'Nota de otros bancos (refuerzo)',
    'EXT_3_pow3': 'Nota de otros bancos (refuerzo fuerte)',
    'EXT_3_pow05': 'Nota de otros bancos (suavizado)',
    'EXT_2_log': 'Nota de otros bancos (escala ajustada)',
    'EXT_3_log': 'Nota de otros bancos (escala ajustada)',
    'EXT_mean_log': 'Promedio de notas externas (escala ajustada)',

    
    # ==================== EXT SOURCE - RATIOS ====================
    'EXT_1d2': 'Comparación entre notas de otros bancos (1 vs 2)',
    'EXT_2d3': 'Comparación entre notas de otros bancos (2 vs 3)',
    'EXT_1d3': 'Comparación entre notas de otros bancos (1 vs 3)',
    'EXT_3d2': 'Comparación entre notas de otros bancos (3 vs 2)',
    
    # ==================== EXT SOURCE - DIFERENCIAS ====================
    'EXT_1m2': 'Diferencia entre notas de otros bancos (1 y 2)',
    'EXT_2m3': 'Diferencia entre notas de otros bancos (2 y 3)',
    'EXT_1m3': 'Diferencia entre notas de otros bancos (1 y 3)',
    'EXT_max_m_min': 'Mayor diferencia entre notas de otros bancos',

    
    # ==================== EXT SOURCE - MEDIAS ESPECIALES ====================
    'EXT_harmonic': 'Promedio conservador de notas de otros bancos',
    'EXT_geometric': 'Promedio equilibrado de notas de otros bancos',
    'EXT_quadratic': 'Promedio que penaliza notas malas de otros bancos',
    
    # ==================== FEATURES DERIVADAS - EDAD ====================
    'AGE_YEARS': 'Edad en años',
    'age': 'Edad',
    'age_sq': 'Edad con mayor peso al aumentar',
    'age_cb': 'Edad con peso muy alto',
    'age_sqrt': 'Edad suavizada',
    'age_log': 'Edad ajustada',
    'age_bin_young': 'Persona joven',
    'age_bin_middle': 'Persona de edad media',
    'age_bin_senior': 'Persona mayor',
    
    # ==================== FEATURES DERIVADAS - EMPLEO ====================
    'YEARS_EMPLOYED': 'Años trabajando',
    'emp_years': 'Tiempo total trabajando',
    'emp_years_sq': 'Tiempo trabajado con mayor peso',
    'emp_years_log': 'Tiempo trabajado ajustado',
    'is_unemployed': 'Actualmente sin trabajo',
    'is_retired': 'Persona jubilada',
    'emp_ratio': 'Proporción de vida trabajada',
    'emp_ratio_sq': 'Proporción de vida trabajada (peso alto)',
    'working_age_ratio': 'Porcentaje de vida laboral activa',
    
    # ==================== RATIOS FINANCIEROS BÁSICOS ====================
    'CREDIT_INCOME_RATIO': 'Crédito comparado con ingresos',
    'ANNUITY_INCOME_RATIO': 'Cuota mensual según ingresos',
    'cr_inc': 'Cuánto crédito pide respecto a lo que gana',
    'an_inc': 'Cuánto paga al mes respecto a lo que gana',
    'cr_an': 'Crédito comparado con la cuota mensual',
    'goods_inc': 'Precio del bien frente a ingresos',
    'goods_cr': 'Relación entre precio y crédito',
    
    # ==================== RATIOS FINANCIEROS - POTENCIAS ====================
    'cr_inc_sq': 'Crédito vs ingresos (impacto alto)',
    'cr_inc_cb': 'Crédito vs ingresos (impacto muy alto)',
    'cr_inc_sqrt': 'Crédito vs ingresos (impacto suavizado)',
    'cr_inc_log': 'Crédito vs ingresos (escala ajustada)',
    'an_inc_sq': 'Cuota vs ingresos (impacto alto)',
    'an_inc_log': 'Cuota vs ingresos (escala ajustada)',
    
    # ==================== TRANSFORMACIONES LOGARÍTMICAS ====================
    'income_log': 'Ingresos ajustados',
    'credit_log': 'Crédito ajustado',
    'annuity_log': 'Cuota mensual ajustada',
    
    # ==================== BUREAU - BÁSICAS ====================
    'bureau_loans': 'Créditos en otros bancos',
    'bureau_days_credit_mean': 'Antigüedad media de los créditos',
    'bureau_days_credit_min': 'Crédito más reciente',
    'bureau_credit_sum': 'Dinero total pedido en otros créditos',
    'bureau_credit_active': 'Créditos activos actualmente',
    'active_loans_count': 'Número de créditos activos',
    'active_debt_sum': 'Deuda total actual',
    'active_overdue_sum': 'Deuda atrasada actual',
    'recent_loans_count': 'Créditos pedidos recientemente',
    'recent_overdue_mean': 'Nivel medio de retrasos recientes',
    
    # ==================== BUREAU - PONDERADAS ====================
    'bureau_weighted_credit': 'Importe de créditos (importancia ajustada)',
    'bureau_weighted_debt': 'Deuda total ajustada',
    'bureau_weighted_overdue': 'Retrasos de pago ajustados',
    'bureau_weighted_active_ratio': 'Proporción de créditos activos (ajustada)',

    
    # ==================== BUREAU - COMPARACIONES TEMPORALES ====================
    'bureau_recent_vs_old_overdue': 'Retrasos recientes frente a antiguos',
    'bureau_recent_vs_old_debt': 'Deuda reciente frente a antigua',
    
    # ==================== BUREAU - DIVERSIFICACIÓN ====================
    'bureau_credit_diversity': 'Variedad de tipos de crédito',
    
    # ==================== BUREAU - RATIOS ====================
    'bureau_debt_to_credit_ratio': 'Qué parte del crédito aún se debe',
    'bureau_overdue_to_debt_ratio': 'Qué parte de la deuda está en retraso',
    'bureau_max_overdue_to_income': 'Mayor retraso de pago comparado con ingresos',

    # ==================== BUREAU - PRÓRROGAS ====================
    'bureau_prolongation_count': 'Veces que pidió más tiempo para pagar',
    'bureau_prolongation_ratio': 'Frecuencia con la que pide prórrogas',
    
    # ==================== BUREAU - MOROSIDAD TEMPORAL ====================
    'bureau_overdue_last_3m': 'Morosidad Últimos 3 Meses',
    'bureau_overdue_last_6m': 'Morosidad Últimos 6 Meses',
    
    # ==================== BUREAU - NUEVOS CRÉDITOS ====================
    'bureau_new_credits_6m': 'Nuevos Créditos 6 Meses',
    'bureau_new_credits_12m': 'Nuevos Créditos 12 Meses',
    
    # ==================== BUREAU - ACELERACIÓN ====================
    'bureau_debt_acceleration': 'Ritmo de aumento de la deuda',
    
    # ==================== BUREAU - CIERRES ====================
    'bureau_closed_last_year': 'Créditos que cerró el último año',
    'bureau_sold_count': 'Créditos vendidos a otros',
    'bureau_bad_debt_count': 'Créditos que no se pudieron cobrar',
    
    # ==================== BUREAU - DPD (DAYS PAST DUE) ====================
    'bureau_dpd_count': 'Días que se ha retrasado en pagar',
    'bureau_severe_dpd_count': 'Retrasos graves en los pagos',
    'bureau_dpd_ratio': 'Porcentaje de pagos con retraso',
    'bureau_recent_dpd': 'Retrasos recientes en los pagos',
    'bureau_current_status_C': 'Estado actual: crédito cerrado',

    # ==================== TARJETAS DE CRÉDITO ====================
    'cc_loans': 'Número de tarjetas',
    'cc_balance_mean': 'Dinero que debe en las tarjetas (media)',
    'cc_limit_mean': 'Límite de las tarjetas (media)',
    'cc_utilization': 'Cuánto usa las tarjetas',
    'cc_util_recent': 'Cuánto ha usado las tarjetas últimamente',
    'cc_balance_trend': 'Si la deuda de las tarjetas sube o baja',
    'cc_max_utilization': 'Uso máximo de las tarjetas',
    'cc_min_payment_ratio': 'Solo paga el mínimo de la tarjeta',
    'cc_drawings_atm_ratio': 'Cuánto dinero saca del cajero con la tarjeta',
    'cc_drawings_count': 'Veces que saca dinero con la tarjeta',
    'cc_dpd_count': 'Retrasos en pagos de la tarjeta',
    'cc_dpd_recent': 'Retrasos recientes en la tarjeta',
    'cc_receivable_ratio': 'Parte de la deuda que aún debe',
    
    # ==================== CUOTAS (INSTALLMENTS) ====================
    'inst_count_total': 'Número total de cuotas',
    'inst_late_ratio_total': 'Cuántas cuotas se pagaron tarde',
    'inst_dbd_mean_total': 'Cuántos días antes suele pagar',
    'inst_late_ratio_1y': 'Cuántas cuotas pagó tarde en el último año',
    'inst_dbd_mean_1y': 'Cuántos días antes suele pagar (último año)',
    'inst_amt_paid_1y': 'Dinero pagado en el último año',
    'inst_late_old': 'Pagos tarde hace mucho tiempo',
    'inst_late_mid': 'Pagos tarde hace un tiempo',
    'inst_late_recent': 'Pagos tarde recientes',
    'inst_partial_payment_ratio': 'Veces que pagó solo una parte',
    'inst_overpayment_ratio': 'Veces que pagó de más',
    'inst_severe_late_ratio': 'Retrasos graves en las cuotas',
    'inst_severe_late_count': 'Número de retrasos graves',
    'inst_max_payment_gap': 'Mayor retraso al pagar',
    
    # ==================== POS CASH ====================
    'pos_loans': 'Número de préstamos',
    'pos_months': 'Cuántos meses lleva pagando',
    'pos_dpd_mean': 'Cuántos días se retrasa al pagar',
    'pos_dpd_def_mean': 'Cuántos días se retrasa del todo',
    'pos_recent_max_dpd': 'Mayor retraso reciente',
    'pos_recent_count_dpd': 'Retrasos recientes al pagar',
    'pos_dpd_mean_3m': 'Retrasos al pagar en los últimos 3 meses',
    'pos_dpd_max_3m': 'Mayor retraso en los últimos 3 meses',
    'pos_dpd_mean_6m': 'Retrasos al pagar en los últimos 6 meses',
    'pos_dpd_max_6m': 'Mayor retraso en los últimos 6 meses',
    'pos_dpd_mean_12m': 'Retrasos al pagar en el último año',
    'pos_dpd_max_12m': 'Mayor retraso en el último año',
    'pos_dpd_acceleration': 'Los retrasos están aumentando',
    'pos_dpd_0_count': 'Pagos sin retraso',
    'pos_dpd_30_60_count': 'Retrasos de 1 a 2 meses',
    'pos_dpd_60_plus_count': 'Retrasos de más de 2 meses',
    'pos_completed_contracts': 'Préstamos ya terminados',
    'pos_active_contracts': 'Préstamos que sigue pagando',
    'pos_dpd_max_ever': 'Mayor retraso que ha tenido',
    
    # ==================== APLICACIONES PREVIAS ====================
    'prev_apps': 'Veces que pidió créditos antes',
    'prev_amt_mean': 'Dinero que solía pedir',
    'prev_amt_max': 'Mayor dinero que pidió',
    'prev_refused': 'Veces que le rechazaron un crédito',
    'prev_approved': 'Veces que le aprobaron un crédito',
    'prev_days_decision_mean': 'Cuántos días tardaban en decidir',
    'prev_weighted_amt': 'Dinero pedido recientemente (más importante)',
    'prev_weighted_refused_ratio': 'Rechazos recientes más importantes',
    'prev_recent_refused_count': 'Rechazos recientes',
    'prev_recent_approved_count': 'Aprobaciones recientes',
    'prev_approval_rate_6m': 'Créditos aprobados en los últimos 6 meses',
    'prev_approval_rate_12m': 'Créditos aprobados en el último año',
    'prev_cancelled_count': 'Créditos que canceló',
    'prev_cancelled_ratio': 'Veces que canceló créditos',
    'prev_amt_approved_ratio': 'Parte del dinero que sí le aprobaron',
    'prev_credit_down_payment_ratio': 'Dinero que puso de entrada',
    'prev_product_diversity': 'Tipos distintos de créditos que ha tenido',
    'prev_revolving_count': 'Créditos que puede usar varias veces',
    'prev_yield_group_high': 'Créditos caros',
    'prev_yield_group_low': 'Créditos baratos',
    'prev_goods_category_count': 'Tipos de cosas que compró con crédito',
    
    # ==================== CROSS FEATURES - DEUDA ====================
    'total_debt_to_income': 'Cuánta deuda tiene comparado con lo que gana',
    'debt_to_income_recent': 'Cuánta deuda reciente tiene comparado con lo que gana',
    'total_overdue_to_income': 'Cuánto debe sin pagar comparado con lo que gana',

    # ==================== CARGA DE PAGO ====================
    'monthly_payment_burden': 'Cuánto le cuesta pagar cada mes',
    'monthly_payment_capacity': 'Cuánto puede pagar al mes sin problemas',

    # ==================== DETERIORO ====================
    'inst_late_deterioration': 'Empeora en pagar tarde',
    'combined_deterioration_score': 'Deterioro reciente',
    'deterioration_composite_score': 'Deterioro acumulado',

    # ==================== RECHAZOS ====================
    'recent_rejection_intensity': 'Muchos rechazos recientes',
    'historical_rejection_rate': 'Rechazos a lo largo del tiempo',

    # ==================== SEÑALES ====================
    'red_flags_count': 'Señales malas',
    'positive_signals_count': 'Señales buenas',

    # ==================== SCORES COMPUESTOS ====================
    'risk_score_composite': 'Riesgo total',
    'credit_mix_score': 'Variedad de créditos',
    'financial_engagement_score': 'Nivel de uso de productos financieros',
    'overall_credit_utilization': 'Cuánto usa sus créditos en total',
    'payment_behavior_score': 'Cómo suele pagar',
    'financial_stability_score': 'Qué tan estable es con el dinero',

    # ==================== INTERACCIONES EXT ====================
    'ext_source_debt_interaction': 'Relación entre nota externa y deuda',
    'ext_source_payment_interaction': 'Relación entre nota externa y pagos',
    'ext_source_risk_interaction': 'Relación entre nota externa y riesgo',

    # ==================== SOLICITUD ====================
    'credit_request_vs_history': 'Lo que pide comparado con su historial',
    'credit_escalation': 'Cada vez pide más dinero',
    'debt_capacity_remaining': 'Cuánto más puede endeudarse',
    'application_risk_score': 'Riesgo de esta solicitud',

    # ==================== EXT x EDAD ====================
    'EXT1_age': 'Nota externa ajustada por edad',
    'EXT2_age': 'Otra nota externa según la edad',
    'EXT3_age': 'Otra nota externa según la edad',
    'EXTm_age': 'Nota externa media según la edad',
    'EXTm_d_age': 'Nota externa comparada con la edad',
    'EXT2_age_sq': 'Efecto fuerte de edad y nota externa',

    # ==================== EXT x EMPLEO ====================
    'EXT2_emp': 'Nota externa según años trabajados',
    'EXTm_emp': 'Nota externa media según trabajo',
    'EXT2_d_emp': 'Nota externa comparada con trabajo',

    # ==================== EXT x FINANZAS ====================
    'EXT2_d_crInc': 'Nota externa comparada con crédito e ingresos',
    'EXTm_d_crInc': 'Nota externa media y crédito',
    'EXT2_x_crInc': 'Nota externa y nivel de deuda',
    'EXTm_x_crInc': 'Nota externa media y deuda',
    'EXT2_d_anInc': 'Nota externa comparada con pagos',
    'EXTm_d_anInc': 'Nota externa media y pagos',
    'EXT2_x_income': 'Nota externa según ingresos',
    'EXTm_x_income': 'Nota externa media según ingresos',

    # ==================== RIESGOS ====================
    'risk_bur': 'Riesgo por deudas externas',
    'risk_bur_ratio': 'Qué tan alto es ese riesgo',
    'risk_inst': 'Riesgo por cuotas',
    'risk_inst_severe': 'Riesgo grave por cuotas',
    'risk_pos': 'Riesgo por préstamos pequeños',
    'risk_cc': 'Riesgo por tarjetas',
    'risk_refused': 'Riesgo por rechazos',
    'risk_score': 'Riesgo general',
    'risk_score_weighted': 'Riesgo general reciente',
    'risk_score_max': 'Mayor riesgo que ha tenido',

    # ==================== EXT x RIESGO ====================
    'EXT2_noRisk': 'Nota externa cuando hay poco riesgo',
    'EXTm_noRisk': 'Nota externa media con poco riesgo',
    'EXT2_d_risk': 'Nota externa comparada con riesgo',
    'EXTm_d_risk': 'Nota externa media comparada con riesgo',
    'cr_inc_riskAdj': 'Crédito e ingresos ajustados al riesgo',
    'an_inc_riskAdj': 'Pagos ajustados al riesgo',

    # ==================== INTERACCIONES ESPECÍFICAS ====================
    'bureau_debt_to_income': 'Deuda comparada con lo que gana',
    'bureau_debt_ratio_ext': 'Deuda y nota externa juntas',
    'diversity_ext': 'Variedad de créditos y nota externa',
    'active_loans_ext': 'Préstamos activos y nota externa',
    'recent_late_ext': 'Pagos tarde recientes y nota externa',
    'dbd_normalized': 'Cuánto suele pagar antes',
    'dbd_ext': 'Pagar antes y nota externa',
    'approval_ext': 'Aprobaciones y nota externa',
    'prev_apps_log': 'Muchas solicitudes anteriores',
    'experience_score': 'Experiencia previa con créditos',
    'cc_util_ext': 'Uso de tarjetas y nota externa',
    'cc_util_risk': 'Uso de tarjetas y riesgo',

    # ==================== SCORES FINALES ====================
    'good_behavior_score': 'Buen comportamiento con el dinero',
    'good_behavior_ext': 'Buen comportamiento y nota externa',
    'net_score': 'Resultado final bueno o malo',
    'net_score_ext': 'Resultado final y nota externa',
    'final_score': 'Resultado final',

}

COLUMN_MAPPING_UI = COLUMN_MAPPING