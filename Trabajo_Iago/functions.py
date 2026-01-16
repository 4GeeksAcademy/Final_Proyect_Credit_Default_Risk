import pandas as pd
import shap
import joblib
import numpy as np
from catboost import Pool


def search_user(sk_id, ammount, credit_type):
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


def predict(user):
    """
    Predicción compatible con XGBoost y CatBoost
    Retorna probabilidades para 0 (No Default) y 1 (Default)
    """
    # Cargar el modelo
    model = joblib.load('../models/catboost_new_dataset.pkl')

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
        model = joblib.load('../models/catboost_new_dataset.pkl')
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