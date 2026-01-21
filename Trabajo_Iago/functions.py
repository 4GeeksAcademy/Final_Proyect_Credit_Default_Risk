import pandas as pd
import shap
import joblib
import numpy as np
from catboost import Pool
import random



def search_user(sk_id, ammount, credit_type):

    """
    uses the training dataset as database to
    find the user and i'ts features
    process them to fit in the models as the
    trining
    returns the full user dataframe with
    the parameters changed
    """

    db = pd.read_csv('../data/processed/home_credit_train_ready.csv')

    user = db.loc[db['SK_ID_CURR'] == sk_id].copy()

    user['AMT_CREDIT'] = ammount

    user['NAME_CONTRACT_TYPE'] = credit_type

    user.replace([np.inf, -np.inf], np.nan, inplace=True)
    user.fillna(0, inplace=True)

    for col in user.select_dtypes("object"):
        user[col] = user[col].replace(0, 'missing')

    for col in user.select_dtypes("object"):
        user[col] = user[col].astype("category")

    user = user.drop(columns=['TARGET', 'SK_ID_CURR'])

    return user

def new_user(**kwargs):

    cols = [
 'NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'OWN_CAR_AGE',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'FLAG_EMAIL',
 'OCCUPATION_TYPE',
 'CNT_FAM_MEMBERS',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'WEEKDAY_APPR_PROCESS_START',
 'HOUR_APPR_PROCESS_START',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'LIVE_CITY_NOT_WORK_CITY',
 'ORGANIZATION_TYPE',
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'DAYS_LAST_PHONE_CHANGE',
 'FLAG_DOCUMENT_2',
 'FLAG_DOCUMENT_3',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_10',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_12',
 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_21',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_REQ_CREDIT_BUREAU_YEAR']

    db = pd.read_csv('../data/processed/home_credit_train_ready.csv')

    # Store original dtypes and categories
    original_dtypes = db.drop(columns=['SK_ID_CURR', 'TARGET']).dtypes.to_dict()
    cat_columns = {}
    for col in db.columns:
        if pd.api.types.is_categorical_dtype(db[col]):
            cat_columns[col] = list(db[col].cat.categories) + ['missing']
        elif db[col].dtype == 'object':
            cat_columns[col] = list(db[col].dropna().unique()) + ['missing']

    # Create empty DataFrame with provided values
    user_data = {col: np.nan for col in original_dtypes.keys()}

    # Fill in provided values
    for key, value in kwargs.items():
        if key in user_data:
            user_data[key] = value

    # Create DataFrame
    user = pd.DataFrame([user_data])

    # Fill NaN values BEFORE converting to categorical
    # Numeric columns get 0, object/categorical columns get 'missing'
    for col in user.columns:
        if col in cat_columns:
            user[col] = user[col].fillna('missing')
        else:
            user[col] = user[col].fillna(0)

    # Now convert to proper dtypes
    for col, dtype in original_dtypes.items():
        if col in cat_columns:
            user[col] = pd.Categorical(user[col], categories=cat_columns[col])
        else:
            try:
                user[col] = user[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    return user.reset_index(drop=True)

def predict(user):
    """
    Predicción compatible con XGBoost y CatBoost
    Retorna probabilidades para 0 (No Default) y 1 (Default)
    """
    # Cargar el modelo
    model = joblib.load('../models/catboost_best_scores.pkl')

    # Detectar tipo de modelo y preparar datos
    model_type = type(model).__name__

    if 'XGB' in model_type or 'Booster' in model_type:
        # Para XGBoost: forzar CPU y limpiar métricas
        try:
            # Configurar el modelo para usar CPU y limpiar métricas problemáticas
            model.set_params(device='cpu', eval_metric=None)
        except:
            try:
                model.set_params(device='cpu')
            except:
                pass



        prediction = model.predict_proba(user)

    elif 'CatBoost' in model_type:
        # Para CatBoost: usar Pool con categorías
        cat_features = user.select_dtypes('category').columns.tolist()
        prediction = model.predict_proba(user)

    else:
        # Fallback genérico
        prediction = model.predict_proba(user)

    return prediction


def explain(user_df: pd.DataFrame):
    """
    Explicación SHAP compatible con XGBoost y CatBoost
    """
    try:
        model = joblib.load('../models/catboost_best_scores.pkl')
        model_type = type(model).__name__

        # Preparar datos según el tipo de modelo
        if 'XGB' in model_type or 'Booster' in model_type:
            # XGBoost: forzar CPU, limpiar métricas y codificar categorías
            try:
                model.set_params(device='cpu', eval_metric=None)
            except:
                try:
                    model.set_params(device='cpu')
                except:
                    pass

            user_encoded = user_df.copy()
            cat_cols = user_encoded.select_dtypes('category').columns
            for col in cat_cols:
                user_encoded[col] = user_encoded[col].cat.codes

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_encoded)

        elif 'CatBoost' in model_type:
            # CatBoost: usar Pool
            cat_features = user_df.select_dtypes('category').columns.tolist()
            pool = Pool(user_df, cat_features=cat_features)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pool)

        else:
            # Fallback
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_df)

        # Procesar SHAP values (clasificación binaria)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Clase 1 (Default)

        shap_values = shap_values[0]  # Solo primer cliente

        # Crear DataFrame de resultados
        result = pd.DataFrame({
            "feature": user_df.columns,
            "value": user_df.iloc[0].values,
            "shap": shap_values
        })

        # Limpiar NaN/Inf
        result.replace([np.inf, -np.inf], 0, inplace=True)
        result.fillna(0, inplace=True)

        # Top 10 features que aumentan y reducen riesgo
        top_bad = result.sort_values("shap", ascending=False).head(10)
        top_good = result.sort_values("shap", ascending=True).head(10)

        return top_bad, top_good

    except Exception as e:
        print(f"Error en explain: {e}")
        # Retornar DataFrames vacíos en caso de error
        empty = pd.DataFrame(columns=["feature", "value", "shap"])
        return empty, empty