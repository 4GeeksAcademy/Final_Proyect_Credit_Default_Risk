import pandas as pd
import numpy as np
import warnings
from sqlalchemy import create_engine, text
from tqdm import tqdm
import gc
import time

warnings.filterwarnings('ignore')

# ============= CARGAR DATASETS =============
print("📥 Cargando datasets...")

application_train = pd.read_csv('../data/interim/aplicationtrainlimpio1.csv')
bureau = pd.read_csv('../data/raw/bureau.csv')
bureau_balance = pd.read_csv('../data/raw/bureau_balance.csv')
previous_application = pd.read_csv('../data/raw/previous_application.csv')
pos_cash_balance = pd.read_csv('../data/raw/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('../data/raw/credit_card_balance.csv')
installments_payments = pd.read_csv('../data/raw/installments_payments.csv')

print("✅ Datasets cargados exitosamente\n")


def create_database_from_dataframes(
        db_connection_string: str = 'sqlite:///home_credit.db'
):
    """
    Crea la base de datos SQL desde los DataFrames ya cargados

    Args:
        db_connection_string: String de conexión a la BD
    """

    print("=" * 60)
    print("📚 CREANDO BASE DE DATOS")
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
            print(f"\n📂 Procesando {table_name}...")
            print(f"   ✅ {len(df):,} registros | {len(df.columns)} columnas")
            print(f"   💾 Guardando en tabla '{table_name}'...")

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
            print(f"   ❌ Error: {str(e)}")

    # Crear índices para optimizar queries
    print("\n🔧 Creando índices...")

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

    print("✅ Índices creados")
    print("\n" + "=" * 60)
    print("✅ BASE DE DATOS CREADA EXITOSAMENTE")
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
    print("🚀 PROCESANDO DATASET COMPLETO PARA ENTRENAMIENTO")
    print("=" * 60)

    engine = create_engine(db_connection_string)

    # 1. Obtener lista de clientes
    print("\n📊 Obteniendo lista de clientes...")
    query = text("""
        SELECT SK_ID_CURR, AMT_CREDIT, NAME_CONTRACT_TYPE, TARGET
        FROM application_train
        ORDER BY SK_ID_CURR
    """)

    clients_info = pd.read_sql(query, engine)
    total_clients = len(clients_info)

    print(f"✅ Total de clientes: {total_clients:,}")

    # 2. Procesar en batches
    pipeline = ClientDataPipelineSQL(db_connection_string)

    all_data = []
    num_batches = (total_clients + batch_size - 1) // batch_size

    print(f"\n⚙️  Procesando en {num_batches} batches...")

    start_time = time.time()
    failed_clients = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_clients)
        batch_clients = clients_info.iloc[batch_start:batch_end]

        print(f"\n📦 Batch {batch_idx + 1}/{num_batches}")

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

        print(f"⏱️  {elapsed / 60:.1f}min | {rate:.1f} clientes/s | Restante: {remaining / 60:.1f}min")

    # 3. Combinar todo
    print("\n🔗 Combinando batches...")
    final_data = pd.concat(all_data, ignore_index=True)

    # 4. Guardar
    print(f"\n💾 Guardando dataset final...")
    final_data.to_parquet(output_path, index=False, compression='snappy')
    final_data.to_csv(output_path.replace('.parquet', '.csv'), index=False)

    # 5. Reporte
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"📊 Clientes procesados: {len(final_data):,}")
    print(f"📊 Total features: {len(final_data.columns):,}")
    print(f"❌ Clientes fallidos: {len(failed_clients)}")
    print(f"⏱️  Tiempo total: {total_time / 60:.1f} minutos")
    print(f"\n🎯 Distribución TARGET:")
    print(final_data['TARGET'].value_counts())
    print(f"\n💾 Archivo guardado: {output_path}")
    print("=" * 60)

    pipeline.close()
    engine.dispose()

    return final_data


# ============= CLASE PIPELINE (copiada de tu código) =============

#%%
## try the object with sql

import pandas as pd
import numpy as np
from typing import Optional
import warnings
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

class ClientDataPipelineSQL:
    """Pipeline optimizado con SQL para cálculos pesados"""

    def __init__(self, db_connection_string: str):
        """
        connection_string ejemplos:
        - SQLite: 'sqlite:///home_credit.db'
        - PostgreSQL: 'postgresql://user:password@localhost/home_credit'
        - MySQL: 'mysql+pymysql://user:password@localhost/home_credit'
        """
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
        client_data = self._create_derived_features(client_data)
        client_data = self._create_cross_features(client_data)

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
        """TODAS las features de bureau en una sola query SQL"""

        query = text("""
        WITH bureau_data AS (
            SELECT
                SK_ID_CURR,
                SK_ID_BUREAU,
                DAYS_CREDIT,
                AMT_CREDIT_SUM,
                AMT_CREDIT_SUM_DEBT,
                AMT_CREDIT_SUM_OVERDUE,
                CREDIT_ACTIVE,
                CREDIT_TYPE,
                CNT_CREDIT_PROLONG,
                EXP(-ABS(DAYS_CREDIT)::float / 365.0) as decay_weight
            FROM bureau
            WHERE SK_ID_CURR = :sk_id_curr
        ),
        bureau_balance_data AS (
            SELECT
                bb.SK_ID_BUREAU,
                COUNT(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 END) as dpd_count,
                COUNT(CASE WHEN bb.STATUS IN ('4','5') THEN 1 END) as severe_dpd_count,
                SUM(CASE WHEN bb.STATUS IN ('4','5') THEN 1 ELSE 0 END)::float /
                    NULLIF(COUNT(*), 0) as dpd_ratio,
                COUNT(CASE WHEN bb.MONTHS_BALANCE >= -6 AND bb.STATUS IN ('1','2','3','4','5') THEN 1 END) as recent_dpd,
                AVG(CASE WHEN bb.MONTHS_BALANCE >= -3 AND bb.STATUS = 'C' THEN 1 ELSE 0 END) as current_status_c
            FROM bureau_balance bb
            GROUP BY bb.SK_ID_BUREAU
        ),
        aggregations AS (
            SELECT
                :sk_id_curr as SK_ID_CURR,

                -- Conteos básicos
                COUNT(DISTINCT SK_ID_BUREAU) as bureau_loans,
                AVG(DAYS_CREDIT) as bureau_days_credit_mean,
                MIN(DAYS_CREDIT) as bureau_days_credit_min,
                SUM(AMT_CREDIT_SUM) as bureau_credit_sum,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) as bureau_credit_active,

                -- Active loans
                COUNT(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 END) as active_loans_count,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_DEBT ELSE 0 END) as active_debt_sum,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_OVERDUE ELSE 0 END) as active_overdue_sum,

                -- Recent loans
                COUNT(CASE WHEN DAYS_CREDIT >= -730 THEN 1 END) as recent_loans_count,
                AVG(CASE WHEN DAYS_CREDIT >= -730 THEN AMT_CREDIT_SUM_OVERDUE ELSE NULL END) as recent_overdue_mean,

                -- Weighted features
                SUM(AMT_CREDIT_SUM * decay_weight) /
                    NULLIF(SUM(decay_weight), 0) as bureau_weighted_credit,
                SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0) * decay_weight) /
                    NULLIF(SUM(decay_weight), 0) as bureau_weighted_debt,
                SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0) * decay_weight) /
                    NULLIF(SUM(decay_weight), 0) as bureau_weighted_overdue,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END::float * decay_weight) /
                    NULLIF(SUM(decay_weight), 0) as bureau_weighted_active_ratio,

                -- Recent vs old comparison
                AVG(CASE WHEN DAYS_CREDIT >= -365 THEN COALESCE(AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) /
                    NULLIF(AVG(CASE WHEN DAYS_CREDIT < -365 THEN COALESCE(AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END), 0) + 1
                    as bureau_recent_vs_old_overdue,
                SUM(CASE WHEN DAYS_CREDIT >= -365 THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN DAYS_CREDIT < -365 THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END), 1)
                    as bureau_recent_vs_old_debt,

                -- Diversity (Shannon entropy)
                -SUM(
                    CASE WHEN credit_type_counts > 0 THEN
                        (credit_type_counts::float / COUNT(*)) *
                        LN(credit_type_counts::float / COUNT(*) + 0.00001)
                    ELSE 0 END
                ) as bureau_credit_diversity,

                -- Critical ratios
                SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 1) as bureau_debt_to_credit_ratio,
                SUM(AMT_CREDIT_SUM_OVERDUE) / NULLIF(SUM(AMT_CREDIT_SUM_DEBT), 1) as bureau_overdue_to_debt_ratio,
                MAX(AMT_CREDIT_SUM_OVERDUE) as bureau_max_overdue_to_income,
                SUM(CASE WHEN CNT_CREDIT_PROLONG > 0 THEN 1 ELSE 0 END) as bureau_prolongation_count,
                AVG(CASE WHEN CNT_CREDIT_PROLONG > 0 THEN 1 ELSE 0 END) as bureau_prolongation_ratio,

                -- Temporal analysis
                AVG(CASE WHEN DAYS_CREDIT >= -90 THEN COALESCE(AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) as bureau_overdue_last_3m,
                AVG(CASE WHEN DAYS_CREDIT >= -180 THEN COALESCE(AMT_CREDIT_SUM_OVERDUE, 0) ELSE NULL END) as bureau_overdue_last_6m,
                COUNT(CASE WHEN DAYS_CREDIT >= -180 THEN 1 END) as bureau_new_credits_6m,
                COUNT(CASE WHEN DAYS_CREDIT >= -365 THEN 1 END) as bureau_new_credits_12m,

                -- Debt acceleration
                AVG(CASE WHEN DAYS_CREDIT >= -180 AND DAYS_CREDIT < -90 THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE NULL END) -
                AVG(CASE WHEN DAYS_CREDIT >= -365 AND DAYS_CREDIT < -180 THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE NULL END)
                    as bureau_debt_acceleration,

                -- Closure and problem statuses
                COUNT(CASE WHEN CREDIT_ACTIVE = 'Closed' AND DAYS_CREDIT >= -365 THEN 1 END) as bureau_closed_last_year,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Sold' THEN 1 ELSE 0 END) as bureau_sold_count,
                SUM(CASE WHEN CREDIT_ACTIVE = 'Bad debt' THEN 1 ELSE 0 END) as bureau_bad_debt_count

            FROM bureau_data
        )
        SELECT
            a.*,
            COALESCE(b.dpd_count, 0) as bureau_dpd_count,
            COALESCE(b.severe_dpd_count, 0) as bureau_severe_dpd_count,
            COALESCE(b.dpd_ratio, 0) as bureau_dpd_ratio,
            COALESCE(b.recent_dpd, 0) as bureau_recent_dpd,
            COALESCE(b.current_status_c, 0) as bureau_current_status_C
        FROM aggregations a
        LEFT JOIN bureau_balance_data b ON a.SK_ID_CURR = :sk_id_curr
        """)

        try:
            result = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})
            if len(result) > 0:
                # Merge con cliente existente
                for col in result.columns:
                    if col != 'SK_ID_CURR':
                        client_data[col] = result[col].iloc[0]
            else:
                # Valores por defecto
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
        """Credit card features en SQL"""
        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as cc_loans,
            AVG(AMT_BALANCE) as cc_balance_mean,
            AVG(AMT_CREDIT_LIMIT_ACTUAL) as cc_limit_mean,
            AVG(AMT_BALANCE::float / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) as cc_utilization,

            AVG(CASE WHEN MONTHS_BALANCE >= -3
                THEN AMT_BALANCE::float / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)
                ELSE NULL END) as cc_util_recent,

            AVG(CASE WHEN MONTHS_BALANCE >= -3
                THEN AMT_BALANCE ELSE NULL END) -
            AVG(CASE WHEN MONTHS_BALANCE < -3
                THEN AMT_BALANCE ELSE NULL END) as cc_balance_trend,

            MAX(AMT_BALANCE::float / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) as cc_max_utilization,

            SUM(AMT_PAYMENT_CURRENT)::float /
            NULLIF(SUM(AMT_INST_MIN_REGULARITY), 0) as cc_min_payment_ratio,

            SUM(AMT_DRAWINGS_ATM_CURRENT)::float /
            NULLIF(SUM(AMT_DRAWINGS_CURRENT), 1) as cc_drawings_atm_ratio,

            SUM(CNT_DRAWINGS_CURRENT) as cc_drawings_count,
            COUNT(CASE WHEN SK_DPD > 0 THEN 1 END) as cc_dpd_count,
            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as cc_dpd_recent,

            SUM(AMT_RECEIVABLE_PRINCIPAL)::float /
            NULLIF(SUM(AMT_BALANCE), 1) as cc_receivable_ratio,

            STDDEV(AMT_BALANCE) / NULLIF(AVG(AMT_BALANCE), 0) as cc_usage_volatility
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
        """Installments features en SQL"""
        query = text("""
        WITH inst_calc AS (
            SELECT
                SK_ID_CURR,
                DAYS_INSTALMENT,
                DAYS_ENTRY_PAYMENT,
                AMT_INSTALMENT,
                AMT_PAYMENT,
                CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END as late,
                DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT as dbd
            FROM installments_payments
            WHERE SK_ID_CURR = :sk_id_curr
        )
        SELECT
            SK_ID_CURR,
            COUNT(*) as inst_count_total,
            AVG(late) as inst_late_ratio_total,
            AVG(dbd) as inst_dbd_mean_total,

            AVG(CASE WHEN DAYS_INSTALMENT >= -365 THEN late ELSE NULL END) as inst_late_ratio_1y,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365 THEN dbd ELSE NULL END) as inst_dbd_mean_1y,
            SUM(CASE WHEN DAYS_INSTALMENT >= -365 THEN AMT_PAYMENT ELSE 0 END) as inst_amt_paid_1y,

            AVG(CASE WHEN DAYS_INSTALMENT < -365 THEN late ELSE NULL END) as inst_late_old,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365 AND DAYS_INSTALMENT < -180 THEN late ELSE NULL END) as inst_late_mid,
            AVG(CASE WHEN DAYS_INSTALMENT >= -180 THEN late ELSE NULL END) as inst_late_recent,

            AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END) as inst_partial_payment_ratio,
            AVG(CASE WHEN AMT_PAYMENT > AMT_INSTALMENT THEN 1 ELSE 0 END) as inst_overpayment_ratio,
            STDDEV(AMT_PAYMENT) / NULLIF(AVG(AMT_PAYMENT), 0) as inst_payment_volatility,

            AVG(CASE WHEN dbd > 30 THEN 1 ELSE 0 END) as inst_severe_late_ratio,
            COUNT(CASE WHEN dbd > 30 THEN 1 END) as inst_severe_late_count,

            1.0 / (NULLIF(STDDEV(dbd), 0) + 1) as inst_payment_consistency,
            MAX(dbd) - MIN(dbd) as inst_max_payment_gap
        FROM inst_calc
        GROUP BY SK_ID_CURR
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
        """POS cash features en SQL"""
        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as pos_loans,
            COUNT(*) as pos_months,
            AVG(SK_DPD) as pos_dpd_mean,
            AVG(SK_DPD_DEF) as pos_dpd_def_mean,

            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as pos_recent_max_dpd,
            COUNT(CASE WHEN MONTHS_BALANCE >= -6 AND SK_DPD > 0 THEN 1 END) as pos_recent_count_dpd,

            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN SK_DPD ELSE NULL END) as pos_dpd_mean_3m,
            MAX(CASE WHEN MONTHS_BALANCE >= -3 THEN SK_DPD ELSE 0 END) as pos_dpd_max_3m,
            STDDEV(CASE WHEN MONTHS_BALANCE >= -3 THEN SK_DPD ELSE NULL END) as pos_dpd_std_3m,

            AVG(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE NULL END) as pos_dpd_mean_6m,
            MAX(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE 0 END) as pos_dpd_max_6m,
            STDDEV(CASE WHEN MONTHS_BALANCE >= -6 THEN SK_DPD ELSE NULL END) as pos_dpd_std_6m,

            AVG(CASE WHEN MONTHS_BALANCE >= -12 THEN SK_DPD ELSE NULL END) as pos_dpd_mean_12m,
            MAX(CASE WHEN MONTHS_BALANCE >= -12 THEN SK_DPD ELSE 0 END) as pos_dpd_max_12m,
            STDDEV(CASE WHEN MONTHS_BALANCE >= -12 THEN SK_DPD ELSE NULL END) as pos_dpd_std_12m,

            AVG(CASE WHEN MONTHS_BALANCE >= -3 THEN SK_DPD ELSE NULL END) -
            AVG(CASE WHEN MONTHS_BALANCE >= -12 AND MONTHS_BALANCE < -3 THEN SK_DPD ELSE NULL END)
                as pos_dpd_acceleration,

            COUNT(CASE WHEN SK_DPD = 0 THEN 1 END) as pos_dpd_0_count,
            COUNT(CASE WHEN SK_DPD > 30 AND SK_DPD <= 60 THEN 1 END) as pos_dpd_30_60_count,
            COUNT(CASE WHEN SK_DPD > 60 THEN 1 END) as pos_dpd_60_plus_count,

            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 END) as pos_completed_contracts,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 END) as pos_active_contracts,

            STDDEV(SK_DPD) as pos_dpd_volatility,
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
        """Previous applications features en SQL"""
        query = text("""
        WITH prev_calc AS (
            SELECT
                SK_ID_CURR,
                DAYS_DECISION,
                AMT_APPLICATION,
                NAME_CONTRACT_STATUS,
                NAME_CONTRACT_TYPE,
                NAME_YIELD_GROUP,
                NAME_GOODS_CATEGORY,
                AMT_DOWN_PAYMENT,
                EXP(-ABS(DAYS_DECISION)::float / 365.0) as decay_weight
            FROM previous_application
            WHERE SK_ID_CURR = :sk_id_curr
        )
        SELECT
            SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as prev_apps,
            AVG(AMT_APPLICATION) as prev_amt_mean,
            MAX(AMT_APPLICATION) as prev_amt_max,

            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 END) as prev_refused,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) as prev_approved,
            AVG(DAYS_DECISION) as prev_days_decision_mean,

            SUM(AMT_APPLICATION * decay_weight) / NULLIF(SUM(decay_weight), 0) as prev_weighted_amt,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END::float * decay_weight) /
                NULLIF(SUM(decay_weight), 0) as prev_weighted_refused_ratio,

            COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Refused' THEN 1 END) as prev_recent_refused_count,
            COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) as prev_recent_approved_count,

            COUNT(CASE WHEN DAYS_DECISION >= -180 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN DAYS_DECISION >= -180 THEN 1 END), 0) as prev_approval_rate_6m,

            COUNT(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved' THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN DAYS_DECISION >= -365 THEN 1 END), 0) as prev_approval_rate_12m,

            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Cancelled' THEN 1 END) as prev_cancelled_count,
            AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Cancelled' THEN 1 ELSE 0 END) as prev_cancelled_ratio,

            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN AMT_APPLICATION ELSE 0 END)::float /
                NULLIF(SUM(AMT_APPLICATION), 1) as prev_amt_approved_ratio,

            SUM(AMT_DOWN_PAYMENT)::float / NULLIF(SUM(AMT_APPLICATION), 1) as prev_credit_down_payment_ratio,

            COUNT(DISTINCT NAME_CONTRACT_TYPE) as prev_product_diversity,
            COUNT(CASE WHEN NAME_CONTRACT_TYPE = 'Revolving loans' THEN 1 END) as prev_revolving_count,

            COUNT(CASE WHEN NAME_YIELD_GROUP = 'high' THEN 1 END) as prev_yield_group_high,
            COUNT(CASE WHEN NAME_YIELD_GROUP = 'low' THEN 1 END) as prev_yield_group_low,
            COUNT(DISTINCT NAME_GOODS_CATEGORY) as prev_goods_category_count
        FROM prev_calc
        GROUP BY SK_ID_CURR
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

    # ============= MÉTODOS DE DERIVED Y CROSS FEATURES =============
    def _create_derived_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """Crea features derivadas (igual que la versión pandas)"""

        if 'AMT_INCOME_TOTAL' in client_data.columns:
            income = client_data['AMT_INCOME_TOTAL'].replace(0, np.nan).fillna(
                client_data['AMT_INCOME_TOTAL'].median()
            )
            client_data['CREDIT_INCOME_RATIO'] = client_data['AMT_CREDIT'] / income

            if 'AMT_ANNUITY' in client_data.columns:
                client_data['ANNUITY_INCOME_RATIO'] = client_data['AMT_ANNUITY'] / income

        if 'DAYS_BIRTH' in client_data.columns:
            client_data['AGE_YEARS'] = (-client_data['DAYS_BIRTH'] / 365).round()

        if 'DAYS_EMPLOYED' in client_data.columns:
            client_data['YEARS_EMPLOYED'] = (-client_data['DAYS_EMPLOYED'] / 365).clip(lower=0).round()

        # EXT_SOURCE features
        ext_cols = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
                    if col in client_data.columns]

        if len(ext_cols) > 0:
            client_data['EXT_SOURCE_MEAN'] = client_data[ext_cols].mean(axis=1, skipna=True)
            client_data['EXT_SOURCE_MEDIAN'] = client_data[ext_cols].median(axis=1, skipna=True)
            client_data['EXT_SOURCE_STD'] = client_data[ext_cols].std(axis=1, skipna=True)
            client_data['EXT_SOURCE_MIN'] = client_data[ext_cols].min(axis=1, skipna=True)
            client_data['EXT_SOURCE_MAX'] = client_data[ext_cols].max(axis=1, skipna=True)
            client_data['EXT_SOURCE_SUM'] = client_data[ext_cols].sum(axis=1, skipna=True)
            client_data['EXT_SOURCE_PROD'] = client_data[ext_cols].prod(axis=1, skipna=True)
            client_data['EXT_SOURCE_RANGE'] = client_data['EXT_SOURCE_MAX'] - client_data['EXT_SOURCE_MIN']
            client_data['EXT_SOURCE_COUNT'] = client_data[ext_cols].notna().sum(axis=1)

            # Ponderada
            weights = {'EXT_SOURCE_1': 1, 'EXT_SOURCE_2': 2, 'EXT_SOURCE_3': 3}
            total_weight = sum(weights[col] for col in ext_cols if col in client_data.columns)

            if total_weight > 0:
                weighted_sum = sum(
                    client_data[col].fillna(0) * weights[col]
                    for col in ext_cols if col in client_data.columns
                )
                client_data['EXT_SOURCE_WEIGHTED'] = weighted_sum / total_weight

            # Interacciones
            if 'AMT_CREDIT' in client_data.columns:
                client_data['EXT_CREDIT_RATIO'] = client_data['AMT_CREDIT'] * client_data['EXT_SOURCE_MEAN']

            if 'AMT_INCOME_TOTAL' in client_data.columns:
                client_data['EXT_INCOME_RATIO'] = client_data['AMT_INCOME_TOTAL'] * client_data['EXT_SOURCE_MEAN']

            # Diferencias
            if 'EXT_SOURCE_1' in ext_cols and 'EXT_SOURCE_2' in ext_cols:
                client_data['EXT_SOURCE_1_2_DIFF'] = client_data['EXT_SOURCE_1'] - client_data['EXT_SOURCE_2']
                client_data['EXT_SOURCE_1_2_RATIO'] = client_data['EXT_SOURCE_1'] / (client_data['EXT_SOURCE_2'] + 1e-10)

            if 'EXT_SOURCE_2' in ext_cols and 'EXT_SOURCE_3' in ext_cols:
                client_data['EXT_SOURCE_2_3_DIFF'] = client_data['EXT_SOURCE_2'] - client_data['EXT_SOURCE_3']
                client_data['EXT_SOURCE_2_3_RATIO'] = client_data['EXT_SOURCE_2'] / (client_data['EXT_SOURCE_3'] + 1e-10)

        return client_data

    def _create_cross_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """Cross-features (igual que pandas, pero optimizado)"""

        def safe_get(col, default=0):
            return client_data.get(col, pd.Series([default], index=client_data.index)).fillna(default).values[0]

        # Deuda vs ingresos
        client_data['total_debt_to_income'] = (
            (safe_get('bureau_weighted_debt') + safe_get('cc_balance_mean')) /
            (safe_get('AMT_INCOME_TOTAL', 1) + 1)
        )

        client_data['debt_to_income_recent'] = (
            safe_get('bureau_weighted_debt') / (safe_get('AMT_INCOME_TOTAL', 1) + 1)
        )

        # Carga de pago
        client_data['monthly_payment_burden'] = (
            (safe_get('AMT_ANNUITY') + safe_get('inst_amt_paid_1y', 0) / 12) /
            (safe_get('AMT_INCOME_TOTAL', 1) / 12 + 1)
        )

        # Scores de deterioro
        client_data['combined_deterioration_score'] = (
            safe_get('inst_late_deterioration') +
            safe_get('pos_dpd_acceleration') +
            safe_get('cc_util_deterioration', 0) +
            safe_get('bureau_recent_vs_old_overdue', 1) - 1
        )

        client_data['deterioration_composite_score'] = (
            safe_get('inst_late_deterioration') * 2 +
            safe_get('cc_balance_growing', 0) * 1.5 +
            (safe_get('bureau_recent_vs_old_debt', 1) - 1) +
            safe_get('pos_dpd_acceleration')
        )

        # Intensidad de rechazos
        client_data['recent_rejection_intensity'] = (
            safe_get('prev_recent_refused_count') /
            (safe_get('prev_recent_refused_count') + safe_get('prev_recent_approved_count') + 1)
        )

        # Capacidad de pago mensual
        client_data['monthly_payment_capacity'] = (
            (safe_get('AMT_INCOME_TOTAL', 1) / 12) -
            (safe_get('AMT_ANNUITY') + safe_get('inst_amt_paid_1y', 0) / 12)
        )

        # Red flags
        client_data['red_flags_count'] = (
            (safe_get('bureau_bad_debt_count') > 0).astype(int) +
            (safe_get('bureau_sold_count') > 0).astype(int) +
            (safe_get('prev_cancelled_ratio') > 0.3).astype(int) +
            (safe_get('inst_severe_late_ratio') > 0.2).astype(int) +
            (safe_get('pos_dpd_60_plus_count') > 0).astype(int) +
            (safe_get('cc_dpd_recent') > 30).astype(int)
        )

        # Señales positivas
        client_data['positive_signals_count'] = (
            (safe_get('inst_overpayment_ratio') > 0).astype(int) +
            (safe_get('bureau_current_status_C') > 0.5).astype(int) +
            (safe_get('pos_completion_ratio') > 0.5).astype(int) +
            (safe_get('prev_approval_rate_12m') > 0.7).astype(int)
        )

        # Credit mix y engagement
        client_data['credit_mix_score'] = (
            safe_get('bureau_credit_diversity') +
            safe_get('prev_product_diversity', 0) / 5
        )

        client_data['financial_engagement_score'] = (
            np.log1p(safe_get('bureau_loans')) +
            np.log1p(safe_get('prev_apps')) +
            np.log1p(safe_get('cc_loans'))
        )

        return client_data

    # ============= VALORES POR DEFECTO =============
    @staticmethod
    def _get_default_bureau_values():
        return {
            'bureau_loans': 0, 'bureau_days_credit_mean': 0,
            'bureau_days_credit_min': 0, 'bureau_credit_sum': 0,
            'bureau_credit_active': 0, 'active_loans_count': 0,
            'active_debt_sum': 0, 'active_overdue_sum': 0,
            'recent_loans_count': 0, 'recent_overdue_mean': 0,
            'bureau_weighted_credit': 0, 'bureau_weighted_debt': 0,
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
            'bureau_dpd_ratio': 0, 'bureau_recent_dpd': 0,
            'bureau_current_status_C': 0
        }

    @staticmethod
    def _get_default_cc_values():
        return {
            'cc_loans': 0, 'cc_balance_mean': 0, 'cc_limit_mean': 0,
            'cc_utilization': 0, 'cc_util_recent': 0, 'cc_util_historical': 0,
            'cc_balance_trend': 0, 'cc_util_deterioration': 0,
            'cc_max_utilization': 0, 'cc_min_payment_ratio': 0,
            'cc_balance_growing': 0, 'cc_drawings_atm_ratio': 0,
            'cc_drawings_count': 0, 'cc_dpd_count': 0,
            'cc_dpd_recent': 0, 'cc_receivable_ratio': 0,
            'cc_usage_volatility': 0
        }

    @staticmethod
    def _get_default_installments_values():
        return {
            'inst_count_total': 0, 'inst_late_ratio_total': 0,
            'inst_dbd_mean_total': 0, 'inst_late_ratio_1y': 0,
            'inst_dbd_mean_1y': 0, 'inst_amt_paid_1y': 0,
            'inst_late_trend': 0, 'inst_late_old': 0,
            'inst_late_mid': 0, 'inst_late_recent': 0,
            'inst_late_deterioration': 0, 'inst_partial_payment_ratio': 0,
            'inst_overpayment_ratio': 0, 'inst_payment_volatility': 0,
            'inst_severe_late_ratio': 0, 'inst_severe_late_count': 0,
            'inst_payment_improvement': 0, 'inst_payment_consistency': 0,
            'inst_max_payment_gap': 0, 'inst_last_3_late_ratio': 0
        }

    @staticmethod
    def _get_default_pos_values():
        return {
            'pos_loans': 0, 'pos_months': 0, 'pos_dpd_mean': 0,
            'pos_dpd_def_mean': 0, 'pos_recent_max_dpd': 0,
            'pos_recent_count_dpd': 0, 'pos_dpd_mean_3m': 0,
            'pos_dpd_max_3m': 0, 'pos_dpd_std_3m': 0,
            'pos_dpd_mean_6m': 0, 'pos_dpd_max_6m': 0,
            'pos_dpd_std_6m': 0, 'pos_dpd_mean_12m': 0,
            'pos_dpd_max_12m': 0, 'pos_dpd_std_12m': 0,
            'pos_dpd_acceleration': 0, 'pos_dpd_0_count': 0,
            'pos_dpd_30_60_count': 0, 'pos_dpd_60_plus_count': 0,
            'pos_dpd_worsening': 0, 'pos_completed_contracts': 0,
            'pos_active_contracts': 0, 'pos_completion_ratio': 0,
            'pos_dpd_volatility': 0, 'pos_dpd_max_ever': 0
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
            'prev_amt_increasing_trend': 1, 'prev_product_diversity': 0,
            'prev_revolving_count': 0, 'prev_yield_group_high': 0,
            'prev_yield_group_low': 0, 'prev_goods_category_count': 0
        }

    def close(self):
        """Cierra la conexión"""
        self.conn.close()



# ============= EJECUCIÓN SIMPLE =============

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

    print("\n🎉 ¡TODO LISTO PARA ENTRENAR!")