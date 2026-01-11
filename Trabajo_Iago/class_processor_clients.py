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

class ClientDataPipelineSQL:
    """Pipeline optimizado con SQL para cálculos pesados"""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.conn = self.engine.connect()

    def get_client_data(self, sk_id_curr: int, amt_credit: float,
                        credit_type: str) -> pd.DataFrame:
        """Obtiene TODOS los datos del cliente desde SQL"""

        client_base = self._get_base_features_sql(sk_id_curr, amt_credit, credit_type)
        if client_base is None or len(client_base) == 0:
            return None

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
                               credit_type: str) -> pd.DataFrame:
        query = text("SELECT * FROM application_train WHERE SK_ID_CURR = :sk_id_curr")
        client = pd.read_sql(query, self.conn, params={'sk_id_curr': sk_id_curr})

        if len(client) == 0:
            return None

        client['AMT_CREDIT'] = amt_credit
        client['NAME_CONTRACT_TYPE'] = credit_type
        return client

    def _add_bureau_features_sql(self, client_data: pd.DataFrame, sk_id_curr: int) -> pd.DataFrame:
        query = text("""
        WITH bureau_data AS (
            SELECT
                SK_ID_CURR, SK_ID_BUREAU, DAYS_CREDIT, AMT_CREDIT_SUM,
                AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_OVERDUE,
                CREDIT_ACTIVE, CREDIT_TYPE, CNT_CREDIT_PROLONG,
                EXP(-ABS(DAYS_CREDIT) / 365.0) as decay_weight
            FROM bureau WHERE SK_ID_CURR = :sk_id_curr
        )
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_BUREAU) as bureau_loans,
            AVG(DAYS_CREDIT) as bureau_days_credit_mean,
            MIN(DAYS_CREDIT) as bureau_days_credit_min,
            SUM(AMT_CREDIT_SUM) as bureau_credit_sum,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) as bureau_credit_active,
            COUNT(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 END) as active_loans_count,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN AMT_CREDIT_SUM_DEBT ELSE 0 END) as active_debt_sum,
            CAST(SUM(AMT_CREDIT_SUM_DEBT) AS REAL) / NULLIF(SUM(AMT_CREDIT_SUM), 1) as bureau_debt_to_credit_ratio,
            CAST(SUM(AMT_CREDIT_SUM * decay_weight) AS REAL) / NULLIF(SUM(decay_weight), 0) as bureau_weighted_credit,
            CAST(SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0) * decay_weight) AS REAL) / NULLIF(SUM(decay_weight), 0) as bureau_weighted_debt
        FROM bureau_data
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
        except:
            defaults = self._get_default_bureau_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_credit_card_features_sql(self, client_data: pd.DataFrame, sk_id_curr: int) -> pd.DataFrame:
        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as cc_loans,
            AVG(AMT_BALANCE) as cc_balance_mean,
            AVG(AMT_CREDIT_LIMIT_ACTUAL) as cc_limit_mean,
            AVG(CAST(AMT_BALANCE AS REAL) / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) as cc_utilization,
            MAX(CAST(AMT_BALANCE AS REAL) / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) as cc_max_utilization,
            COUNT(CASE WHEN SK_DPD > 0 THEN 1 END) as cc_dpd_count
        FROM credit_card_balance
        WHERE SK_ID_CURR = :sk_id_curr
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
        except:
            defaults = self._get_default_cc_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_installments_features_sql(self, client_data: pd.DataFrame, sk_id_curr: int) -> pd.DataFrame:
        query = text("""
        WITH inst_calc AS (
            SELECT
                SK_ID_CURR,
                CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END as late,
                DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT as dbd,
                AMT_PAYMENT
            FROM installments_payments
            WHERE SK_ID_CURR = :sk_id_curr
        )
        SELECT
            SK_ID_CURR,
            COUNT(*) as inst_count_total,
            AVG(late) as inst_late_ratio_total,
            AVG(dbd) as inst_dbd_mean_total,
            SUM(AMT_PAYMENT) as inst_amt_paid_total
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
        except:
            defaults = self._get_default_installments_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_pos_cash_features_sql(self, client_data: pd.DataFrame, sk_id_curr: int) -> pd.DataFrame:
        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV) as pos_loans,
            AVG(SK_DPD) as pos_dpd_mean,
            MAX(SK_DPD) as pos_dpd_max,
            COUNT(CASE WHEN SK_DPD > 0 THEN 1 END) as pos_dpd_count
        FROM pos_cash_balance
        WHERE SK_ID_CURR = :sk_id_curr
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
        except:
            defaults = self._get_default_pos_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _add_previous_app_features_sql(self, client_data: pd.DataFrame, sk_id_curr: int) -> pd.DataFrame:
        query = text("""
        SELECT
            :sk_id_curr as SK_ID_CURR,
            COUNT(*) as prev_apps,
            AVG(AMT_APPLICATION) as prev_amt_mean,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 END) as prev_refused,
            COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) as prev_approved
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
        except:
            defaults = self._get_default_prev_values()
            for col, val in defaults.items():
                client_data[col] = val

        return client_data

    def _create_derived_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        if 'AMT_INCOME_TOTAL' in client_data.columns:
            income = client_data['AMT_INCOME_TOTAL'].replace(0, np.nan).fillna(
                client_data['AMT_INCOME_TOTAL'].median()
            )
            client_data['CREDIT_INCOME_RATIO'] = client_data['AMT_CREDIT'] / income

        if 'DAYS_BIRTH' in client_data.columns:
            client_data['AGE_YEARS'] = (-client_data['DAYS_BIRTH'] / 365).round()

        ext_cols = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
                    if col in client_data.columns]

        if len(ext_cols) > 0:
            client_data['EXT_SOURCE_MEAN'] = client_data[ext_cols].mean(axis=1, skipna=True)

        return client_data

    def _create_cross_features(self, client_data: pd.DataFrame) -> pd.DataFrame:
        def safe_get(col, default=0):
            return client_data.get(col, pd.Series([default], index=client_data.index)).fillna(default).values[0]

        client_data['total_debt_to_income'] = (
                (safe_get('bureau_weighted_debt') + safe_get('cc_balance_mean')) /
                (safe_get('AMT_INCOME_TOTAL', 1) + 1)
        )

        return client_data

    @staticmethod
    def _get_default_bureau_values():
        return {'bureau_loans': 0, 'bureau_days_credit_mean': 0, 'bureau_days_credit_min': 0,
                'bureau_credit_sum': 0, 'bureau_credit_active': 0, 'active_loans_count': 0,
                'active_debt_sum': 0, 'bureau_debt_to_credit_ratio': 0,
                'bureau_weighted_credit': 0, 'bureau_weighted_debt': 0}

    @staticmethod
    def _get_default_cc_values():
        return {'cc_loans': 0, 'cc_balance_mean': 0, 'cc_limit_mean': 0,
                'cc_utilization': 0, 'cc_max_utilization': 0, 'cc_dpd_count': 0}

    @staticmethod
    def _get_default_installments_values():
        return {'inst_count_total': 0, 'inst_late_ratio_total': 0,
                'inst_dbd_mean_total': 0, 'inst_amt_paid_total': 0}

    @staticmethod
    def _get_default_pos_values():
        return {'pos_loans': 0, 'pos_dpd_mean': 0, 'pos_dpd_max': 0, 'pos_dpd_count': 0}

    @staticmethod
    def _get_default_prev_values():
        return {'prev_apps': 0, 'prev_amt_mean': 0, 'prev_refused': 0, 'prev_approved': 0}

    def close(self):
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