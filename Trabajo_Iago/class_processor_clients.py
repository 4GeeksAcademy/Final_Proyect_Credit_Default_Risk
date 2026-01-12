import pandas as pd
import numpy as np
import warnings
from sqlalchemy import create_engine, text
import joblib
from tqdm import tqdm
import gc
import time
import shap

warnings.filterwarnings('ignore')

# ============= CARGAR DATASETS =============

application_train = pd.read_csv('../data/interim/aplicationtrainlimpio1.csv')
bureau = pd.read_csv('../data/raw/bureau.csv')
bureau_balance = pd.read_csv('../data/raw/bureau_balance.csv')
previous_application = pd.read_csv('../data/raw/previous_application.csv')
pos_cash_balance = pd.read_csv('../data/raw/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('../data/raw/credit_card_balance.csv')
installments_payments = pd.read_csv('../data/raw/installments_payments.csv')




def create_database_from_dataframes(
        db_connection_string: str = 'sqlite:///home_credit.db'
):
    """
    Crea la base de datos SQL desde los DataFrames ya cargados

    Args:
        db_connection_string: String de conexión a la BD
    """

    print("=" * 60)
    print("CREANDO BASE DE DATOS")
    print("=" * 60)

    engine = create_engine(db_connection_string)

    # Mapeo de DataFrames a tablas
    dataframes = {
        'application_train': application_train,
        'bureau': bureau,
        'bureau_balance': bureau_balance,
        'credit_card_balance': credit_card_balance,
        'installments_payments': installments_payments,
        'pos_cash_balance': pos_cash_balance,
        'previous_application': previous_application
    }

    for table_name, df in dataframes.items():
        try:
            print(f"\nProcesando {table_name}...")
            print(f"   {len(df):,} registros | {len(df.columns)} columnas")
            print(f"   Guardando en tabla '{table_name}'...")

            # Guardar en SQL
            df.to_sql(
                table_name,
                engine,
                if_exists='replace',
                index=False,
                chunksize=10000
            )

            print(f"   ✅ Tabla '{table_name}' creada exitosamente")

        except Exception as e:
            print(f"Error: {str(e)}")

    # Crear índices para optimizar queries
    print("\nCreando índices...")

    with engine.connect() as conn:
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_app_sk ON application_train(SK_ID_CURR)",
            "CREATE INDEX IF NOT EXISTS idx_bureau_sk ON bureau(SK_ID_CURR)",
            "CREATE INDEX IF NOT EXISTS idx_bureau_bureau ON bureau(SK_ID_BUREAU)",
            "CREATE INDEX IF NOT EXISTS idx_bb_bureau ON bureau_balance(SK_ID_BUREAU)",
            "CREATE INDEX IF NOT EXISTS idx_cc_sk ON credit_card_balance(SK_ID_CURR)",
            "CREATE INDEX IF NOT EXISTS idx_inst_sk ON installments_payments(SK_ID_CURR)",
            "CREATE INDEX IF NOT EXISTS idx_pos_sk ON pos_cash_balance(SK_ID_CURR)",
            "CREATE INDEX IF NOT EXISTS idx_prev_sk ON previous_application(SK_ID_CURR)"
        ]

        for idx_query in indices:
            try:
                conn.execute(text(idx_query))
                conn.commit()
            except:
                pass

    print("Índices creados")
    print("\n" + "=" * 60)
    print("BASE DE DATOS CREADA EXITOSAMENTE")
    print("=" * 60)

    engine.dispose()


def process_full_dataset_for_training(
        db_connection_string: str = 'sqlite:///home_credit.db',
        output_path: str = 'home_credit_train_ready.parquet',
        batch_size: int = 1000
):
    """
    Procesa TODO el dataset y lo deja listo para entrenar

    Args:
        db_connection_string: Conexión a la BD
        output_path: Donde guardar el dataset final
        batch_size: Clientes por batch

    Returns:
        DataFrame con todos los datos procesados
    """

    print("\n" + "=" * 60)
    print("PROCESANDO DATASET COMPLETO PARA ENTRENAMIENTO")
    print("=" * 60)

    engine = create_engine(db_connection_string)

    # 1. Obtener lista de clientes
    print("\nObteniendo lista de clientes...")
    query = text("""
        SELECT SK_ID_CURR, AMT_CREDIT, NAME_CONTRACT_TYPE, TARGET
        FROM application_train
        ORDER BY SK_ID_CURR
    """)

    clients_info = pd.read_sql(query, engine)
    total_clients = len(clients_info)

    print(f"Total de clientes: {total_clients:,}")

    # 2. Procesar en batches
    pipeline = ClientDataPipelineSQL(db_connection_string)

    all_data = []
    num_batches = (total_clients + batch_size - 1) // batch_size

    print(f"\nProcesando en {num_batches} batches...")

    start_time = time.time()
    failed_clients = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_clients)
        batch_clients = clients_info.iloc[batch_start:batch_end]

        print(f"\nBatch {batch_idx + 1}/{num_batches}")

        batch_data = []

        for idx, row in tqdm(batch_clients.iterrows(),
                             total=len(batch_clients),
                             desc=f"Procesando"):

            try:
                client_data = pipeline.get_client_data(
                    sk_id_curr=row['SK_ID_CURR'],
                    amt_credit=row['AMT_CREDIT'],
                    credit_type=row['NAME_CONTRACT_TYPE']
                )

                if client_data is not None and len(client_data) > 0:
                    # Agregar TARGET
                    client_data['TARGET'] = row['TARGET']
                    batch_data.append(client_data)
                else:
                    failed_clients.append(row['SK_ID_CURR'])

            except Exception as e:
                failed_clients.append(row['SK_ID_CURR'])
                continue

        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            all_data.append(batch_df)

        del batch_data
        gc.collect()

        # Progreso
        elapsed = time.time() - start_time
        processed = batch_end
        rate = processed / elapsed
        remaining = (total_clients - processed) / rate if rate > 0 else 0

        print(f"{elapsed / 60:.1f}min | {rate:.1f} clientes/s | Restante: {remaining / 60:.1f}min")

    # 3. Combinar todo
    print("\nCombinando batches...")
    final_data = pd.concat(all_data, ignore_index=True)

    # 4. Guardar
    print(f"\nGuardando dataset final...")
    final_data.to_parquet(output_path, index=False, compression='snappy')
    final_data.to_csv(output_path.replace('.parquet', '.csv'), index=False)

    # 5. Reporte
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Clientes procesados: {len(final_data):,}")
    print(f"Total features: {len(final_data.columns):,}")
    print(f"Clientes fallidos: {len(failed_clients)}")
    print(f"⏱Tiempo total: {total_time / 60:.1f} minutos")
    print(f"\nDistribución TARGET:")
    print(final_data['TARGET'].value_counts())
    print(f"\nArchivo guardado: {output_path}")
    print("=" * 60)

    pipeline.close()
    engine.dispose()

    return final_data


def process_client(
    sk_id_curr: int,
    new_income: float,
    new_credit_type: str,
    db_connection_string: str = "sqlite:///home_credit.db"
) -> pd.DataFrame:
    """
    Genera un escenario contrafactual completo para un cliente:
    - Busca el cliente en application_train
    - Sustituye AMT_INCOME_TOTAL y NAME_CONTRACT_TYPE
    - Ejecuta toda la pipeline SQL
    - Devuelve el dataframe listo para el modelo
    """

    # Cargar cliente base
    base = application_train.loc[
        application_train["SK_ID_CURR"] == sk_id_curr
    ].copy()

    if base.empty:
        raise ValueError(f"Cliente {sk_id_curr} no existe")

    # Aplicar contrafactual
    base["AMT_CREDIT"] = new_income
    base["NAME_CONTRACT_TYPE"] = new_credit_type

    # Extraer datos necesarios para el pipeline
    amt_credit = float(base["AMT_CREDIT"].iloc[0])
    credit_type = base["NAME_CONTRACT_TYPE"].iloc[0]

    # Ejecutar pipeline SQL completo
    pipeline = ClientDataPipelineSQL(db_connection_string)

    engineered = pipeline.get_client_data(
        sk_id_curr=sk_id_curr,
        amt_credit=amt_credit,
        credit_type=credit_type
    )

    if engineered is None:
        raise RuntimeError("El pipeline no devolvió datos")

    # Inyectar el nuevo income (porque SQL no lo conoce)
    engineered["AMT_CREDIT"] = new_income

    engineered = engineered.drop(columns=['SK_ID_CURR', 'TARGET'])

    return engineered


def predict(user):
    """
    usando los datos creados en process client
    crea una doble prediccion en porcentaje
    para 0 (No Default)
    y 1 (Default)
    """

    #loads the model
    model = joblib.load("../models/catboost_best_scores.pkl")

    #% for 0(no default) and 1(default) in that order
    prediction = model.predict_proba(user)

    return prediction

def explain(user_df: pd.DataFrame):
    """
    Devuelve explicación SHAP de un cliente:
    - top 10 features que suben riesgo
    - top 10 que lo reducen
    """

    #Loads the model
    model = joblib.load("../models/catboost_best_scores.pkl")

    # Explainer
    explainer = shap.TreeExplainer(model)

    #SHAP values
    shap_values = explainer.shap_values(user_df)

    #Binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]  # solo un cliente

    #df for results
    result = pd.DataFrame({
        "feature": user_df.columns,
        "value": user_df.iloc[0].values,
        "shap": shap_values
    })

    #Top drivers
    top_bad = result.sort_values("shap", ascending=False).head(10)
    top_good = result.sort_values("shap", ascending=True).head(10)

    return top_bad, top_good

# ============= CLASE PIPELINE =============

## try the object with sql

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from typing import Optional


class ClientDataPipelineSQL:
    """Pipeline optimizado con SQL para cálculos pesados - SQLite Compatible"""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.conn = self.engine.connect()

    def get_client_data(self, sk_id_curr: int, amt_credit: float,
                        credit_type: str) -> Optional[pd.DataFrame]:
        """Obtiene TODOS los datos del cliente desde SQL"""

        # 1. Base features
        client_base = self._get_base_features_sql(sk_id_curr, amt_credit, credit_type)
        if client_base is None or len(client_base) == 0:
            return None

        # 2. Agregar features de todas las tablas
        client_data = client_base.copy()
        client_data = self._add_bureau_features_sql(client_data, sk_id_curr)
        client_data = self._add_credit_card_features_sql(client_data, sk_id_curr)
        client_data = self._add_installments_features_sql(client_data, sk_id_curr)
        client_data = self._add_pos_cash_features_sql(client_data, sk_id_curr)
        client_data = self._add_previous_app_features_sql(client_data, sk_id_curr)

        # 3. Derived y cross features
        client_data = self._create_derived_features(client_data)
        client_data = self._create_cross_features(client_data)

        # 4. Complete features (EXT combinations, risk scores, etc.)
        client_data = self._create_complete_features(client_data)

        return client_data

    def _get_base_features_sql(self, sk_id_curr: int, amt_credit: float,
                               credit_type: str) -> Optional[pd.DataFrame]:
        """Obtiene features base desde SQL"""
        query = text("""
            SELECT *
            FROM application_train
            WHERE SK_ID_CURR = :sk_id_curr
        """)

        client = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})

        if len(client) == 0:
            return None

        client['AMT_CREDIT'] = amt_credit
        client['NAME_CONTRACT_TYPE'] = credit_type
        return client

    def _add_bureau_features_sql(self, client_data: pd.DataFrame,
                                 sk_id_curr: int) -> pd.DataFrame:
        """Bureau features - SQLite Compatible"""

        query = text("""
        WITH bureau_balance_agg AS (
            SELECT
                b.SK_ID_BUREAU,
                COUNT(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 END) as dpd_count,
                COUNT(CASE WHEN bb.STATUS IN ('4','5') THEN 1 END) as severe_dpd_count,
                CAST(SUM(CASE WHEN bb.STATUS IN ('4','5') THEN 1 ELSE 0 END) AS REAL) /
                    MAX(COUNT(*), 1) as dpd_ratio,
                COUNT(CASE WHEN bb.MONTHS_BALANCE >= -6 AND bb.STATUS IN ('1','2','3','4','5') THEN 1 END) as recent_dpd,
                AVG(CASE WHEN bb.MONTHS_BALANCE >= -3 AND bb.STATUS = 'C' THEN 1.0 ELSE 0.0 END) as current_status_c
            FROM bureau b
            LEFT JOIN bureau_balance bb ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
            WHERE b.SK_ID_CURR = :sk_id_curr
            GROUP BY b.SK_ID_BUREAU
        )
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT b.SK_ID_BUREAU) as bureau_loans,
            AVG(b.DAYS_CREDIT) as bureau_days_credit_mean,
            MIN(b.DAYS_CREDIT) as bureau_days_credit_min,
            SUM(b.AMT_CREDIT_SUM) as bureau_credit_sum,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) as bureau_credit_active,
            COUNT(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 END) as active_loans_count,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN COALESCE(b.AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END) as active_debt_sum,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0) ELSE 0 END) as active_overdue_sum,
            COUNT(CASE WHEN b.DAYS_CREDIT >= -730 THEN 1 END) as recent_loans_count,
            AVG(CASE WHEN b.DAYS_CREDIT >= -730 THEN b.AMT_CREDIT_SUM_OVERDUE ELSE NULL END) as recent_overdue_mean,
            AVG(COALESCE(b.AMT_CREDIT_SUM, 0)) as bureau_weighted_credit,
            AVG(COALESCE(b.AMT_CREDIT_SUM_DEBT, 0)) as bureau_weighted_debt,
            AVG(COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0)) as bureau_weighted_overdue,
            AVG(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END) as bureau_weighted_active_ratio,
            COALESCE(
                AVG(CASE WHEN b.DAYS_CREDIT >= -365 THEN COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) /
                NULLIF(AVG(CASE WHEN b.DAYS_CREDIT < -365 THEN COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END), 0),
                1.0
            ) as bureau_recent_vs_old_overdue,
            COALESCE(
                SUM(CASE WHEN b.DAYS_CREDIT >= -365 THEN COALESCE(b.AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END) /
                NULLIF(SUM(CASE WHEN b.DAYS_CREDIT < -365 THEN COALESCE(b.AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END), 0),
                1.0
            ) as bureau_recent_vs_old_debt,
            COUNT(DISTINCT b.CREDIT_TYPE) as bureau_credit_diversity,
            COALESCE(CAST(SUM(b.AMT_CREDIT_SUM_DEBT) AS REAL) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0), 0) as bureau_debt_to_credit_ratio,
            COALESCE(CAST(SUM(b.AMT_CREDIT_SUM_OVERDUE) AS REAL) / NULLIF(SUM(b.AMT_CREDIT_SUM_DEBT), 0), 0) as bureau_overdue_to_debt_ratio,
            MAX(COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0)) as bureau_max_overdue_to_income,
            SUM(CASE WHEN b.CNT_CREDIT_PROLONG > 0 THEN 1 ELSE 0 END) as bureau_prolongation_count,
            AVG(CASE WHEN b.CNT_CREDIT_PROLONG > 0 THEN 1.0 ELSE 0.0 END) as bureau_prolongation_ratio,
            AVG(CASE WHEN b.DAYS_CREDIT >= -90 THEN COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) as bureau_overdue_last_3m,
            AVG(CASE WHEN b.DAYS_CREDIT >= -180 THEN COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) as bureau_overdue_last_6m,
            COUNT(CASE WHEN b.DAYS_CREDIT >= -180 THEN 1 END) as bureau_new_credits_6m,
            COUNT(CASE WHEN b.DAYS_CREDIT >= -365 THEN 1 END) as bureau_new_credits_12m,
            AVG(CASE WHEN b.DAYS_CREDIT >= -180 AND b.DAYS_CREDIT < -90 THEN COALESCE(b.AMT_CREDIT_SUM_DEBT, 0) ELSE NULL END) -
            AVG(CASE WHEN b.DAYS_CREDIT >= -365 AND b.DAYS_CREDIT < -180 THEN COALESCE(b.AMT_CREDIT_SUM_DEBT, 0) ELSE NULL END) as bureau_debt_acceleration,
            COUNT(CASE WHEN b.CREDIT_ACTIVE = 'Closed' AND b.DAYS_CREDIT >= -365 THEN 1 END) as bureau_closed_last_year,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Sold' THEN 1 ELSE 0 END) as bureau_sold_count,
            SUM(CASE WHEN b.CREDIT_ACTIVE = 'Bad debt' THEN 1 ELSE 0 END) as bureau_bad_debt_count,
            COALESCE(SUM(bba.dpd_count), 0) as bureau_dpd_count,
            COALESCE(SUM(bba.severe_dpd_count), 0) as bureau_severe_dpd_count,
            COALESCE(AVG(bba.dpd_ratio), 0) as bureau_dpd_ratio,
            COALESCE(SUM(bba.recent_dpd), 0) as bureau_recent_dpd,
            COALESCE(AVG(bba.current_status_c), 0) as bureau_current_status_C
        FROM bureau b
        LEFT JOIN bureau_balance_agg bba ON b.SK_ID_BUREAU = bba.SK_ID_BUREAU
        WHERE b.SK_ID_CURR = :sk_id_curr
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].iloc[0]
            else:
                defaults = self._get_default_bureau_values()
                for col, val in defaults.items():
                    client_data[col] = val
        except Exception as e:
            print(f"Error en bureau features: {e}")
            defaults = self._get_default_bureau_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_credit_card_features_sql(self, client_data: pd.DataFrame,
                                      sk_id_curr: int) -> pd.DataFrame:
        """Credit card features - SQLite Compatible"""

        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as cc_loans,
            AVG(AMT_BALANCE) as cc_balance_mean,
            AVG(AMT_CREDIT_LIMIT_ACTUAL) as cc_limit_mean,
            AVG(CAST(AMT_BALANCE AS REAL) / MAX(AMT_CREDIT_LIMIT_ACTUAL, 1)) as cc_utilization,
            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN CAST(AMT_BALANCE AS REAL) / MAX(AMT_CREDIT_LIMIT_ACTUAL, 1) ELSE NULL END) as cc_util_recent,
            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN AMT_BALANCE ELSE NULL END) -
            AVG(CASE WHEN MONTHS_BALANCE < -3 THEN AMT_BALANCE ELSE NULL END) as cc_balance_trend,
            MAX(CAST(AMT_BALANCE AS REAL) / MAX(AMT_CREDIT_LIMIT_ACTUAL, 1)) as cc_max_utilization,
            CAST(SUM(AMT_PAYMENT_CURRENT) AS REAL) / MAX(SUM(AMT_INST_MIN_REGULARITY), 1) as cc_min_payment_ratio,
            CAST(SUM(AMT_DRAWINGS_ATM_CURRENT) AS REAL) / MAX(SUM(AMT_DRAWINGS_CURRENT), 1) as cc_drawings_atm_ratio,
            SUM(CNT_DRAWINGS_CURRENT) as cc_drawings_count,
            COUNT(CASE WHEN SK_DPD > 0 THEN 1 END) as cc_dpd_count,
            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as cc_dpd_recent,
            CAST(SUM(AMT_RECEIVABLE_PRINCIPAL) AS REAL) / MAX(SUM(AMT_BALANCE), 1) as cc_receivable_ratio
        FROM credit_card_balance
        WHERE SK_ID_CURR = :sk_id_curr
        GROUP BY SK_ID_CURR
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].fillna(0).iloc[0]
            else:
                defaults = self._get_default_cc_values()
                for col, val in defaults.items():
                    client_data[col] = val
        except Exception as e:
            print(f"Error en CC features: {e}")
            defaults = self._get_default_cc_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_installments_features_sql(self, client_data: pd.DataFrame,
                                       sk_id_curr: int) -> pd.DataFrame:
        """Installments features - SQLite Compatible"""

        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(*) as inst_count_total,
            AVG(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END) as inst_late_ratio_total,
            AVG(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) as inst_dbd_mean_total,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE NULL END) as inst_late_ratio_1y,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365 THEN CAST(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT AS REAL) ELSE NULL END) as inst_dbd_mean_1y,
            SUM(CASE WHEN DAYS_INSTALMENT >= -365 THEN AMT_PAYMENT ELSE 0 END) as inst_amt_paid_1y,
            AVG(CASE WHEN DAYS_INSTALMENT < -365 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE NULL END) as inst_late_old,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365 AND DAYS_INSTALMENT < -180 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE NULL END) as inst_late_mid,
            AVG(CASE WHEN DAYS_INSTALMENT >= -180 AND DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE NULL END) as inst_late_recent,
            AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) as inst_partial_payment_ratio,
            AVG(CASE WHEN AMT_PAYMENT > AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) as inst_overpayment_ratio,
            AVG(CASE WHEN (DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) > 30 THEN 1.0 ELSE 0.0 END) as inst_severe_late_ratio,
            COUNT(CASE WHEN (DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) > 30 THEN 1 END) as inst_severe_late_count,
            MAX(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) - MIN(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) as inst_max_payment_gap
        FROM installments_payments
        WHERE SK_ID_CURR = :sk_id_curr
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].fillna(0).iloc[0]
            else:
                defaults = self._get_default_installments_values()
                for col, val in defaults.items():
                    client_data[col] = val
        except Exception as e:
            print(f"Error en installments features: {e}")
            defaults = self._get_default_installments_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_pos_cash_features_sql(self, client_data: pd.DataFrame,
                                   sk_id_curr: int) -> pd.DataFrame:
        """POS cash features - SQLite Compatible"""

        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as pos_loans,
            COUNT(*) as pos_months,
            AVG(SK_DPD) as pos_dpd_mean,
            AVG(SK_DPD_DEF) as pos_dpd_def_mean,
            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as pos_recent_max_dpd,
            COUNT(CASE WHEN MONTHS_BALANCE >= -6 AND SK_DPD > 0 THEN 1 END) as pos_recent_count_dpd,
            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN CAST(SK_DPD AS REAL) ELSE NULL END) as pos_dpd_mean_3m,
            MAX(CASE WHEN MONTHS_BALANCE >= -3 THEN SK_DPD ELSE 0 END) as pos_dpd_max_3m,
            AVG(CASE WHEN MONTHS_BALANCE >= -6 THEN CAST(SK_DPD AS REAL) ELSE NULL END) as pos_dpd_mean_6m,
            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as pos_dpd_max_6m,
            AVG(CASE WHEN MONTHS_BALANCE >= -12 THEN CAST(SK_DPD AS REAL) ELSE NULL END) as pos_dpd_mean_12m,
            MAX(CASE WHEN MONTHS_BALANCE >= -12 THEN SK_DPD ELSE 0 END) as pos_dpd_max_12m,
            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN CAST(SK_DPD AS REAL) ELSE NULL END) -
            AVG(CASE WHEN MONTHS_BALANCE >= -12 AND MONTHS_BALANCE < -3 THEN CAST(SK_DPD AS REAL) ELSE NULL END) as pos_dpd_acceleration,
            COUNT(CASE WHEN SK_DPD = 0 THEN 1 END) as pos_dpd_0_count,
            COUNT(CASE WHEN SK_DPD > 30 AND SK_DPD <= 60 THEN 1 END) as pos_dpd_30_60_count,
            COUNT(CASE WHEN SK_DPD > 60 THEN 1 END) as pos_dpd_60_plus_count,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 END) as pos_completed_contracts,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 END) as pos_active_contracts,
            MAX(SK_DPD) as pos_dpd_max_ever
        FROM pos_cash_balance
        WHERE SK_ID_CURR = :sk_id_curr
        GROUP BY SK_ID_CURR
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].fillna(0).iloc[0]
            else:
                defaults = self._get_default_pos_values()
                for col, val in defaults.items():
                    client_data[col] = val
        except Exception as e:
            print(f"Error en POS features: {e}")
            defaults = self._get_default_pos_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_previous_app_features_sql(self, client_data: pd.DataFrame,
                                       sk_id_curr: int) -> pd.DataFrame:
        """Previous applications features - SQLite Compatible"""

        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as prev_apps,
            AVG(AMT_APPLICATION) as prev_amt_mean,
            MAX(AMT_APPLICATION) as prev_amt_max,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 END) as prev_refused,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) as prev_approved,
            AVG(DAYS_DECISION) as prev_days_decision_mean,
            AVG(AMT_APPLICATION) as prev_weighted_amt,
            AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END) as prev_weighted_refused_ratio,
            COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Refused' THEN 1 END) as prev_recent_refused_count,
            COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) as prev_recent_approved_count,
            CAST(COUNT(CASE WHEN DAYS_DECISION >= -180 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) AS REAL) /
                MAX(COUNT(CASE WHEN DAYS_DECISION >= -180 THEN 1 END), 1) as prev_approval_rate_6m,
            CAST(COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) AS REAL) /
                MAX(COUNT(CASE WHEN DAYS_DECISION >= -365 THEN 1 END), 1) as prev_approval_rate_12m,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Cancelled' THEN 1 END) as prev_cancelled_count,
            AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Cancelled' THEN 1.0 ELSE 0.0 END) as prev_cancelled_ratio,
            CAST(SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN AMT_APPLICATION ELSE 0 END) AS REAL) /
                MAX(SUM(AMT_APPLICATION), 1) as prev_amt_approved_ratio,
            CAST(SUM(AMT_DOWN_PAYMENT) AS REAL) / MAX(SUM(AMT_APPLICATION), 1) as prev_credit_down_payment_ratio,
            COUNT(DISTINCT NAME_CONTRACT_TYPE) as prev_product_diversity,
            COUNT(CASE WHEN NAME_CONTRACT_TYPE = 'Revolving loans' THEN 1 END) as prev_revolving_count,
            COUNT(CASE WHEN NAME_YIELD_GROUP = 'high' THEN 1 END) as prev_yield_group_high,
            COUNT(CASE WHEN NAME_YIELD_GROUP IN ('low_normal', 'low_action') THEN 1 END) as prev_yield_group_low,
            COUNT(DISTINCT NAME_GOODS_CATEGORY) as prev_goods_category_count
        FROM previous_application
        WHERE SK_ID_CURR = :sk_id_curr
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].fillna(0).iloc[0]
            else:
                defaults = self._get_default_prev_values()
                for col, val in defaults.items():
                    client_data[col] = val
        except Exception as e:
            print(f"Error en previous app features: {e}")
            defaults = self._get_default_prev_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _create_derived_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """Crea features derivadas"""
        if 'AMT_INCOME_TOTAL' in client_data.columns:
            income = client_data['AMT_INCOME_TOTAL'].replace(0, np.nan).fillna(
                client_data['AMT_INCOME_TOTAL'].median() if client_data['AMT_INCOME_TOTAL'].median() else 1
            )
            client_data['CREDIT_INCOME_RATIO'] = client_data['AMT_CREDIT'] / income
            if 'AMT_ANNUITY' in client_data.columns:
                client_data['ANNUITY_INCOME_RATIO'] = client_data['AMT_ANNUITY'] / income

        if 'DAYS_BIRTH' in client_data.columns:
            client_data['AGE_YEARS'] = (-client_data['DAYS_BIRTH'] / 365).round()

        if 'DAYS_EMPLOYED' in client_data.columns:
            client_data['YEARS_EMPLOYED'] = (-client_data['DAYS_EMPLOYED'] / 365).clip(lower=0).round()

        ext_cols = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
                    if col in client_data.columns]

        if len(ext_cols) > 0:
            client_data['EXT_SOURCE_MEAN'] = client_data[ext_cols].mean(axis=1, skipna=True)
            client_data['EXT_SOURCE_MIN'] = client_data[ext_cols].min(axis=1, skipna=True)
            client_data['EXT_SOURCE_MAX'] = client_data[ext_cols].max(axis=1, skipna=True)
            client_data['EXT_SOURCE_SUM'] = client_data[ext_cols].sum(axis=1, skipna=True)

        return client_data

    def _create_cross_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """Cross-features entre tablas - VERSIÓN COMPLETA"""

        def safe_get(col, default=0):
            if col in client_data.columns:
                val = client_data[col].iloc[0] if len(client_data) > 0 else default
                return val if pd.notna(val) else default
            return default

        # === DEUDA VS INGRESOS ===
        income = safe_get('AMT_INCOME_TOTAL', 1)

        client_data['total_debt_to_income'] = (
                (safe_get('bureau_weighted_debt') + safe_get('cc_balance_mean') + safe_get('active_debt_sum')) /
                (income + 1)
        )

        client_data['debt_to_income_recent'] = (
                safe_get('bureau_weighted_debt') / (income + 1)
        )

        client_data['total_overdue_to_income'] = (
                (safe_get('bureau_weighted_overdue') + safe_get('active_overdue_sum')) /
                (income + 1)
        )

        # === CARGA DE PAGO MENSUAL ===
        monthly_income = income / 12

        client_data['monthly_payment_burden'] = (
                (safe_get('AMT_ANNUITY') + safe_get('inst_amt_paid_1y', 0) / 12) /
                (monthly_income + 1)
        )

        client_data['monthly_payment_capacity'] = (
                monthly_income -
                (safe_get('AMT_ANNUITY') + safe_get('inst_amt_paid_1y', 0) / 12)
        )

        # === SCORES DE DETERIORO ===
        inst_late_deterioration = (
                safe_get('inst_late_recent') - safe_get('inst_late_old')
        )
        client_data['inst_late_deterioration'] = inst_late_deterioration

        client_data['combined_deterioration_score'] = (
                inst_late_deterioration +
                safe_get('pos_dpd_acceleration') +
                safe_get('cc_balance_trend', 0) / 10000 +  # Normalizar
                (safe_get('bureau_recent_vs_old_overdue', 1) - 1)
        )

        client_data['deterioration_composite_score'] = (
                inst_late_deterioration * 2 +
                (1 if safe_get('cc_balance_trend', 0) > 0 else 0) * 1.5 +
                (safe_get('bureau_recent_vs_old_debt', 1) - 1) +
                safe_get('pos_dpd_acceleration')
        )

        # === HISTORIAL DE RECHAZOS ===
        total_recent_apps = safe_get('prev_recent_refused_count') + safe_get('prev_recent_approved_count')

        client_data['recent_rejection_intensity'] = (
                safe_get('prev_recent_refused_count') / (total_recent_apps + 1)
        )

        client_data['historical_rejection_rate'] = (
                safe_get('prev_refused') / (safe_get('prev_apps') + 1)
        )

        # === RED FLAGS (SEÑALES NEGATIVAS) ===
        client_data['red_flags_count'] = (
                (1 if safe_get('bureau_bad_debt_count') > 0 else 0) +
                (1 if safe_get('bureau_sold_count') > 0 else 0) +
                (1 if safe_get('prev_cancelled_ratio') > 0.3 else 0) +
                (1 if safe_get('inst_severe_late_ratio') > 0.2 else 0) +
                (1 if safe_get('pos_dpd_60_plus_count') > 0 else 0) +
                (1 if safe_get('cc_dpd_recent') > 30 else 0) +
                (1 if safe_get('bureau_overdue_to_debt_ratio') > 0.1 else 0)
        )

        # === SEÑALES POSITIVAS ===
        client_data['positive_signals_count'] = (
                (1 if safe_get('inst_overpayment_ratio') > 0.1 else 0) +
                (1 if safe_get('bureau_current_status_C') > 0.5 else 0) +
                (1 if safe_get('pos_completed_contracts') > safe_get('pos_active_contracts') else 0) +
                (1 if safe_get('prev_approval_rate_12m') > 0.7 else 0) +
                (1 if safe_get('bureau_closed_last_year') > 0 else 0)
        )

        # === SCORE DE RIESGO COMPUESTO ===
        client_data['risk_score_composite'] = (
                client_data['red_flags_count'] * 2 -
                client_data['positive_signals_count'] +
                client_data['combined_deterioration_score']
        )

        # === CREDIT MIX Y DIVERSIFICACIÓN ===
        client_data['credit_mix_score'] = (
                safe_get('bureau_credit_diversity') +
                safe_get('prev_product_diversity', 0) / 3
        )

        client_data['financial_engagement_score'] = (
                np.log1p(safe_get('bureau_loans')) +
                np.log1p(safe_get('prev_apps')) +
                np.log1p(safe_get('cc_loans')) +
                np.log1p(safe_get('pos_loans'))
        )

        # === UTILIZACIÓN DE CRÉDITO ===
        total_credit_limit = safe_get('bureau_credit_sum') + safe_get('cc_limit_mean') * safe_get('cc_loans')
        total_debt = safe_get('bureau_weighted_debt') + safe_get('cc_balance_mean') * safe_get('cc_loans')

        client_data['overall_credit_utilization'] = (
                total_debt / (total_credit_limit + 1)
        )

        # === COMPORTAMIENTO DE PAGO ===
        client_data['payment_behavior_score'] = (
                (1 - safe_get('inst_late_ratio_total')) * 0.4 +
                (1 - safe_get('bureau_dpd_ratio')) * 0.3 +
                safe_get('inst_overpayment_ratio') * 0.3
        )

        # === ESTABILIDAD FINANCIERA ===
        client_data['financial_stability_score'] = (
                safe_get('YEARS_EMPLOYED', 0) / 10 +  # Normalizado
                (1 - safe_get('inst_payment_volatility', 0)) * 0.5 +
                (1 if safe_get('bureau_new_credits_6m') < 2 else 0) * 0.5
        )

        # === INTERACCIONES CON EXT_SOURCE ===
        ext_mean = safe_get('EXT_SOURCE_MEAN', 0.5)

        client_data['ext_source_debt_interaction'] = (
                ext_mean * (1 - client_data['total_debt_to_income'].iloc[0] if len(client_data) > 0 else 0)
        )

        client_data['ext_source_payment_interaction'] = (
            ext_mean * client_data['payment_behavior_score'].iloc[0] if len(client_data) > 0 else 0
        )

        client_data['ext_source_risk_interaction'] = (
            ext_mean / (client_data['risk_score_composite'].iloc[0] + 5) if len(client_data) > 0 else 0
        )

        # === RATIOS DE SOLICITUD VS HISTORIAL ===
        client_data['credit_request_vs_history'] = (
                safe_get('AMT_CREDIT') / (safe_get('prev_amt_mean') + 1)
        )

        client_data['credit_escalation'] = (
                safe_get('AMT_CREDIT') / (safe_get('bureau_weighted_credit') + 1)
        )

        # === CAPACIDAD DE ENDEUDAMIENTO ===
        client_data['debt_capacity_remaining'] = (
                (income * 0.4) -  # Asumiendo 40% máximo de DTI
                (safe_get('AMT_ANNUITY') + safe_get('active_debt_sum') / 12)
        )

        # === SCORE FINAL DE APLICACIÓN ===
        client_data['application_risk_score'] = (
            client_data['risk_score_composite'].iloc[0] if len(client_data) > 0 else 0 +
                                                                                     (1 - ext_mean) * 3 +
                                                                                     client_data[
                                                                                         'total_debt_to_income'].iloc[
                                                                                         0] if len(
                client_data) > 0 else 0 * 2 -
                                      client_data['payment_behavior_score'].iloc[0] if len(client_data) > 0 else 0 * 2
        )

        return client_data

    def _create_complete_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering completo - Todas las combinaciones de EXT_SOURCE,
        ratios financieros, interacciones y scores de riesgo.
        """
        df = client_data.copy()

        # === EXT_SOURCE BASE ===
        ext1 = df.get('EXT_SOURCE_1', pd.Series([0.5] * len(df), index=df.index)).fillna(0.5).replace(0, 0.5)
        ext2 = df.get('EXT_SOURCE_2', pd.Series([0.5] * len(df), index=df.index)).fillna(0.5).replace(0, 0.5)
        ext3 = df.get('EXT_SOURCE_3', pd.Series([0.5] * len(df), index=df.index)).fillna(0.5).replace(0, 0.5)

        # === ESTADÍSTICAS BÁSICAS EXT_SOURCE ===
        df['EXT_mean'] = (ext1 + ext2 + ext3) / 3
        ext_concat = pd.concat([ext1, ext2, ext3], axis=1)
        df['EXT_std'] = ext_concat.std(axis=1)
        df['EXT_min'] = ext_concat.min(axis=1)
        df['EXT_max'] = ext_concat.max(axis=1)
        df['EXT_sum'] = ext1 + ext2 + ext3
        df['EXT_prod'] = ext1 * ext2 * ext3
        df['EXT_range'] = df['EXT_max'] - df['EXT_min']

        # === WEIGHTED COMBINATIONS ===
        for w1, w2, w3 in [(0.1, 0.6, 0.3), (0.05, 0.7, 0.25), (0.15, 0.55, 0.30)]:
            col_name = f'EXT_w_{str(w1).replace(".", "")}_{str(w2).replace(".", "")}_{str(w3).replace(".", "")}'
            df[col_name] = ext1 * w1 + ext2 * w2 + ext3 * w3

        # === PRODUCTOS CRUZADOS ===
        df['EXT_1x2'] = ext1 * ext2
        df['EXT_2x3'] = ext2 * ext3
        df['EXT_1x3'] = ext1 * ext3
        df['EXT_1x2x3'] = ext1 * ext2 * ext3

        # === POTENCIAS ===
        for p in [2, 3, 0.5]:
            p_name = str(p).replace(".", "")
            df[f'EXT_2_pow{p_name}'] = ext2 ** p
            df[f'EXT_mean_pow{p_name}'] = df['EXT_mean'] ** p
            df[f'EXT_3_pow{p_name}'] = ext3 ** p

        # === TRANSFORMACIONES ===
        df['EXT_2_log'] = np.log1p(ext2)
        df['EXT_3_log'] = np.log1p(ext3)
        df['EXT_mean_log'] = np.log1p(df['EXT_mean'])

        # === RATIOS ENTRE EXT_SOURCE ===
        df['EXT_1d2'] = ext1 / (ext2 + 0.001)
        df['EXT_2d3'] = ext2 / (ext3 + 0.001)
        df['EXT_1d3'] = ext1 / (ext3 + 0.001)
        df['EXT_3d2'] = ext3 / (ext2 + 0.001)

        # === DIFERENCIAS ===
        df['EXT_1m2'] = ext1 - ext2
        df['EXT_2m3'] = ext2 - ext3
        df['EXT_1m3'] = ext1 - ext3
        df['EXT_max_m_min'] = df['EXT_max'] - df['EXT_min']

        # === MEDIAS ESPECIALES ===
        df['EXT_harmonic'] = 3 / (1 / (ext1 + 0.01) + 1 / (ext2 + 0.01) + 1 / (ext3 + 0.01))
        df['EXT_geometric'] = (ext1 * ext2 * ext3) ** (1 / 3)
        df['EXT_quadratic'] = np.sqrt((ext1 ** 2 + ext2 ** 2 + ext3 ** 2) / 3)

        # === AGE FEATURES ===
        if 'DAYS_BIRTH' in df.columns:
            df['age'] = -df['DAYS_BIRTH'] / 365.25
            df['age_sq'] = df['age'] ** 2
            df['age_cb'] = df['age'] ** 3
            df['age_sqrt'] = np.sqrt(df['age'].clip(lower=0))
            df['age_log'] = np.log1p(df['age'].clip(lower=0))

            # Age bins
            df['age_bin_young'] = (df['age'] < 30).astype(int)
            df['age_bin_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(int)
            df['age_bin_senior'] = (df['age'] >= 50).astype(int)

        # === EMPLOYMENT FEATURES ===
        if 'DAYS_EMPLOYED' in df.columns:
            days_emp = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['emp_years'] = (-days_emp / 365.25).clip(lower=0)
            df['emp_years_sq'] = df['emp_years'] ** 2
            df['emp_years_log'] = np.log1p(df['emp_years'].fillna(0))
            df['is_unemployed'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            df['is_retired'] = (
                        (df['DAYS_EMPLOYED'] == 365243) & (df.get('age', pd.Series([0] * len(df))) > 55)).astype(int)

        # === EMPLOYMENT RATIO ===
        if 'age' in df.columns and 'emp_years' in df.columns:
            df['emp_ratio'] = df['emp_years'].fillna(0) / (df['age'] + 0.01)
            df['emp_ratio_sq'] = df['emp_ratio'] ** 2
            df['working_age_ratio'] = df['emp_years'].fillna(0) / (df['age'] - 18).clip(lower=1)

        # === FINANCIAL RATIOS BASE ===
        income = df.get('AMT_INCOME_TOTAL', pd.Series([1] * len(df), index=df.index)).replace(0, np.nan)
        income = income.fillna(income.median() if income.median() > 0 else 1)

        credit = df.get('AMT_CREDIT', pd.Series([1] * len(df), index=df.index)).replace(0, np.nan)
        credit = credit.fillna(credit.median() if credit.median() > 0 else 1)

        annuity = df.get('AMT_ANNUITY', pd.Series([1] * len(df), index=df.index)).replace(0, np.nan)
        annuity = annuity.fillna(annuity.median() if annuity.median() > 0 else 1)

        goods_price = df.get('AMT_GOODS_PRICE', pd.Series([1] * len(df), index=df.index)).replace(0, np.nan)
        goods_price = goods_price.fillna(goods_price.median() if goods_price.median() > 0 else credit)

        # === RATIOS PRINCIPALES ===
        df['cr_inc'] = (credit / income).clip(upper=50)
        df['an_inc'] = (annuity / income).clip(upper=5)
        df['cr_an'] = (credit / annuity).clip(upper=100)
        df['goods_inc'] = (goods_price / income).clip(upper=50)
        df['goods_cr'] = (goods_price / credit).clip(upper=5)

        # === POTENCIAS DE RATIOS ===
        df['cr_inc_sq'] = df['cr_inc'] ** 2
        df['cr_inc_cb'] = df['cr_inc'] ** 3
        df['cr_inc_sqrt'] = np.sqrt(df['cr_inc'])
        df['cr_inc_log'] = np.log1p(df['cr_inc'])

        df['an_inc_sq'] = df['an_inc'] ** 2
        df['an_inc_log'] = np.log1p(df['an_inc'])

        # === INCOME FEATURES ===
        df['income_log'] = np.log1p(income)
        df['credit_log'] = np.log1p(credit)
        df['annuity_log'] = np.log1p(annuity)

        # === INTERACTIONS: EXT x AGE ===
        if 'age' in df.columns:
            df['EXT1_age'] = ext1 * df['age']
            df['EXT2_age'] = ext2 * df['age']
            df['EXT3_age'] = ext3 * df['age']
            df['EXTm_age'] = df['EXT_mean'] * df['age']
            df['EXTm_d_age'] = df['EXT_mean'] / (df['age'] + 0.01)
            df['EXT2_age_sq'] = ext2 * df['age_sq']

        # === INTERACTIONS: EXT x EMPLOYMENT ===
        if 'emp_years' in df.columns:
            df['EXT2_emp'] = ext2 * df['emp_years'].fillna(0)
            df['EXTm_emp'] = df['EXT_mean'] * df['emp_years'].fillna(0)
            df['EXT2_d_emp'] = ext2 / (df['emp_years'].fillna(0) + 0.01)

        # === INTERACTIONS: EXT x FINANCIAL RATIOS ===
        df['EXT2_d_crInc'] = ext2 / (df['cr_inc'] + 0.01)
        df['EXTm_d_crInc'] = df['EXT_mean'] / (df['cr_inc'] + 0.01)
        df['EXT2_x_crInc'] = ext2 * df['cr_inc']
        df['EXTm_x_crInc'] = df['EXT_mean'] * df['cr_inc']

        df['EXT2_d_anInc'] = ext2 / (df['an_inc'] + 0.01)
        df['EXTm_d_anInc'] = df['EXT_mean'] / (df['an_inc'] + 0.01)

        df['EXT2_x_income'] = ext2 * income / 100000  # Normalizado
        df['EXTm_x_income'] = df['EXT_mean'] * income / 100000

        # === RISK SCORE COMPONENTS ===
        risk_parts = []
        risk_weights = []

        if 'bureau_dpd_count' in df.columns:
            df['risk_bur'] = df['bureau_dpd_count'].clip(upper=20) / 20
            risk_parts.append(df['risk_bur'])
            risk_weights.append(0.3)

        if 'bureau_dpd_ratio' in df.columns:
            df['risk_bur_ratio'] = df['bureau_dpd_ratio'].clip(upper=1)
            risk_parts.append(df['risk_bur_ratio'])
            risk_weights.append(0.2)

        if 'inst_late_ratio_total' in df.columns:
            df['risk_inst'] = df['inst_late_ratio_total'].clip(upper=1)
            risk_parts.append(df['risk_inst'])
            risk_weights.append(0.25)

        if 'inst_severe_late_ratio' in df.columns:
            df['risk_inst_severe'] = df['inst_severe_late_ratio'].clip(upper=1)
            risk_parts.append(df['risk_inst_severe'])
            risk_weights.append(0.15)

        if 'pos_dpd_mean' in df.columns:
            df['risk_pos'] = df['pos_dpd_mean'].clip(upper=30) / 30
            risk_parts.append(df['risk_pos'])
            risk_weights.append(0.2)

        if 'cc_dpd_count' in df.columns:
            df['risk_cc'] = df['cc_dpd_count'].clip(upper=20) / 20
            risk_parts.append(df['risk_cc'])
            risk_weights.append(0.15)

        if 'prev_weighted_refused_ratio' in df.columns:
            df['risk_refused'] = df['prev_weighted_refused_ratio'].clip(upper=1)
            risk_parts.append(df['risk_refused'])
            risk_weights.append(0.1)

        # === COMPOSITE RISK SCORES ===
        if len(risk_parts) > 0:
            # Simple average
            df['risk_score'] = sum(risk_parts) / len(risk_parts)

            # Weighted average (normalize weights)
            total_weight = sum(risk_weights[:len(risk_parts)])
            if total_weight > 0:
                df['risk_score_weighted'] = sum(
                    p * w for p, w in zip(risk_parts, risk_weights)
                ) / total_weight
            else:
                df['risk_score_weighted'] = df['risk_score']

            # Max risk
            df['risk_score_max'] = pd.concat(risk_parts, axis=1).max(axis=1)

            # === EXT x RISK INTERACTIONS ===
            df['EXT2_noRisk'] = ext2 * (1 - df['risk_score'])
            df['EXTm_noRisk'] = df['EXT_mean'] * (1 - df['risk_score'])
            df['EXT2_d_risk'] = ext2 / (df['risk_score'] + 0.01)
            df['EXTm_d_risk'] = df['EXT_mean'] / (df['risk_score'] + 0.01)

            # Risk adjusted credit
            df['cr_inc_riskAdj'] = df['cr_inc'] * (1 + df['risk_score'])
            df['an_inc_riskAdj'] = df['an_inc'] * (1 + df['risk_score'])

        else:
            # Default values if no risk components
            df['risk_score'] = 0.5
            df['risk_score_weighted'] = 0.5
            df['risk_score_max'] = 0.5
            df['EXT2_noRisk'] = ext2 * 0.5
            df['EXTm_noRisk'] = df['EXT_mean'] * 0.5
            df['EXT2_d_risk'] = ext2 / 0.51
            df['EXTm_d_risk'] = df['EXT_mean'] / 0.51
            df['cr_inc_riskAdj'] = df['cr_inc'] * 1.5
            df['an_inc_riskAdj'] = df['an_inc'] * 1.5

        # === BUREAU SPECIFIC INTERACTIONS ===
        if 'bureau_weighted_debt' in df.columns:
            df['bureau_debt_to_income'] = df['bureau_weighted_debt'] / (income + 1)
            df['bureau_debt_ratio_ext'] = df['bureau_debt_to_income'] * ext2

        if 'bureau_credit_diversity' in df.columns:
            df['diversity_ext'] = df['bureau_credit_diversity'] * df['EXT_mean']

        if 'active_loans_count' in df.columns:
            df['active_loans_ext'] = df['active_loans_count'] * ext2

        # === INSTALLMENTS INTERACTIONS ===
        if 'inst_late_ratio_1y' in df.columns:
            df['recent_late_ext'] = df['inst_late_ratio_1y'] * (1 - ext2)

        if 'inst_dbd_mean_1y' in df.columns:
            df['dbd_normalized'] = df['inst_dbd_mean_1y'].clip(lower=-30, upper=30) / 30
            df['dbd_ext'] = df['dbd_normalized'] * ext2

        # === PREVIOUS APPLICATIONS INTERACTIONS ===
        if 'prev_approval_rate_12m' in df.columns:
            df['approval_ext'] = df['prev_approval_rate_12m'] * ext2

        if 'prev_apps' in df.columns:
            df['prev_apps_log'] = np.log1p(df['prev_apps'])
            df['experience_score'] = df['prev_apps_log'] * df['EXT_mean']

        # === CREDIT CARD INTERACTIONS ===
        if 'cc_utilization' in df.columns:
            df['cc_util_ext'] = df['cc_utilization'].clip(upper=2) * (1 - ext2)
            df['cc_util_risk'] = df['cc_utilization'].clip(upper=2) * df.get('risk_score', 0.5)

        # === COMPOSITE SCORES ===
        # Good behavior score
        good_parts = []
        if 'inst_overpayment_ratio' in df.columns:
            good_parts.append(df['inst_overpayment_ratio'].clip(upper=1))
        if 'prev_approval_rate_12m' in df.columns:
            good_parts.append(df['prev_approval_rate_12m'])
        if 'bureau_current_status_C' in df.columns:
            good_parts.append(df['bureau_current_status_C'])

        if len(good_parts) > 0:
            df['good_behavior_score'] = sum(good_parts) / len(good_parts)
            df['good_behavior_ext'] = df['good_behavior_score'] * ext2
        else:
            df['good_behavior_score'] = 0.5
            df['good_behavior_ext'] = 0.5 * ext2

        # Net score (good - risk)
        df['net_score'] = df['good_behavior_score'] - df.get('risk_score', 0.5)
        df['net_score_ext'] = df['net_score'] * ext2

        # === FINAL COMPOSITE ===
        df['final_score'] = (
                df['EXT_mean'] * 0.4 +
                (1 - df.get('risk_score', 0.5)) * 0.3 +
                df['good_behavior_score'] * 0.2 +
                (1 - df['cr_inc'].clip(upper=10) / 10) * 0.1
        )

        return df

    @staticmethod
    def _get_default_bureau_values():
        return {
            'bureau_loans': 0, 'bureau_days_credit_mean': 0, 'bureau_days_credit_min': 0,
            'bureau_credit_sum': 0, 'bureau_credit_active': 0, 'active_loans_count': 0,
            'active_debt_sum': 0, 'active_overdue_sum': 0, 'recent_loans_count': 0,
            'recent_overdue_mean': 0, 'bureau_weighted_credit': 0, 'bureau_weighted_debt': 0,
            'bureau_weighted_overdue': 0, 'bureau_weighted_active_ratio': 0,
            'bureau_recent_vs_old_overdue': 1, 'bureau_recent_vs_old_debt': 1,
            'bureau_credit_diversity': 0, 'bureau_debt_to_credit_ratio': 0,
            'bureau_overdue_to_debt_ratio': 0, 'bureau_max_overdue_to_income': 0,
            'bureau_prolongation_count': 0, 'bureau_prolongation_ratio': 0,
            'bureau_overdue_last_3m': 0, 'bureau_overdue_last_6m': 0,
            'bureau_new_credits_6m': 0, 'bureau_new_credits_12m': 0,
            'bureau_debt_acceleration': 0, 'bureau_closed_last_year': 0,
            'bureau_sold_count': 0, 'bureau_bad_debt_count': 0,
            'bureau_dpd_count': 0, 'bureau_severe_dpd_count': 0,
            'bureau_dpd_ratio': 0, 'bureau_recent_dpd': 0, 'bureau_current_status_C': 0
        }

    @staticmethod
    def _get_default_cc_values():
        return {
            'cc_loans': 0, 'cc_balance_mean': 0, 'cc_limit_mean': 0,
            'cc_utilization': 0, 'cc_util_recent': 0, 'cc_balance_trend': 0,
            'cc_max_utilization': 0, 'cc_min_payment_ratio': 0,
            'cc_drawings_atm_ratio': 0, 'cc_drawings_count': 0,
            'cc_dpd_count': 0, 'cc_dpd_recent': 0, 'cc_receivable_ratio': 0
        }

    @staticmethod
    def _get_default_installments_values():
        return {
            'inst_count_total': 0, 'inst_late_ratio_total': 0, 'inst_dbd_mean_total': 0,
            'inst_late_ratio_1y': 0, 'inst_dbd_mean_1y': 0, 'inst_amt_paid_1y': 0,
            'inst_late_old': 0, 'inst_late_mid': 0, 'inst_late_recent': 0,
            'inst_partial_payment_ratio': 0, 'inst_overpayment_ratio': 0,
            'inst_severe_late_ratio': 0, 'inst_severe_late_count': 0,
            'inst_max_payment_gap': 0
        }

    @staticmethod
    def _get_default_pos_values():
        return {
            'pos_loans': 0, 'pos_months': 0, 'pos_dpd_mean': 0, 'pos_dpd_def_mean': 0,
            'pos_recent_max_dpd': 0, 'pos_recent_count_dpd': 0, 'pos_dpd_mean_3m': 0,
            'pos_dpd_max_3m': 0, 'pos_dpd_mean_6m': 0, 'pos_dpd_max_6m': 0,
            'pos_dpd_mean_12m': 0, 'pos_dpd_max_12m': 0, 'pos_dpd_acceleration': 0,
            'pos_dpd_0_count': 0, 'pos_dpd_30_60_count': 0, 'pos_dpd_60_plus_count': 0,
            'pos_completed_contracts': 0, 'pos_active_contracts': 0, 'pos_dpd_max_ever': 0
        }

    @staticmethod
    def _get_default_prev_values():
        return {
            'prev_apps': 0, 'prev_amt_mean': 0, 'prev_amt_max': 0,
            'prev_refused': 0, 'prev_approved': 0, 'prev_days_decision_mean': 0,
            'prev_weighted_amt': 0, 'prev_weighted_refused_ratio': 0,
            'prev_recent_refused_count': 0, 'prev_recent_approved_count': 0,
            'prev_approval_rate_6m': 0, 'prev_approval_rate_12m': 0,
            'prev_cancelled_count': 0, 'prev_cancelled_ratio': 0,
            'prev_amt_approved_ratio': 0, 'prev_credit_down_payment_ratio': 0,
            'prev_product_diversity': 0, 'prev_revolving_count': 0,
            'prev_yield_group_high': 0, 'prev_yield_group_low': 0,
            'prev_goods_category_count': 0
        }

    def close(self):
        """Cierra la conexión"""
        self.conn.close()


'''
if __name__ == "__main__":
    # PASO 1: Crear base de datos desde los DataFrames ya cargados
    create_database_from_dataframes(
        db_connection_string='sqlite:///home_credit.db'
    )

    # PASO 2: Procesar dataset completo
    final_data = process_full_dataset_for_training(
        db_connection_string='sqlite:///home_credit.db',
        output_path='../data/processed/home_credit_train_ready.parquet',
        batch_size=1000
    )

    print("\n¡TODO LISTO PARA ENTRENAR!")
'''