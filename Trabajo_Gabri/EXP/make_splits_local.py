import os
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "data/external/application_limpio/applicationtrainlimpio2.csv"
SPLIT_DIR = "data/splits"

RANDOM_STATE = 42
VAL_SIZE = 0.2

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_path = f"{SPLIT_DIR}/train_ids.csv"
    val_path   = f"{SPLIT_DIR}/val_ids.csv"

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("✅ Splits ya existen. No se regeneran.")
        return

    df = pd.read_csv(CSV_PATH, usecols=["SK_ID_CURR", "TARGET"])
    df = df.drop_duplicates(subset="SK_ID_CURR")

    train_ids, val_ids = train_test_split(
        df,
        test_size=VAL_SIZE,
        stratify=df["TARGET"],
        random_state=RANDOM_STATE
    )

    train_ids[["SK_ID_CURR"]].to_csv(train_path, index=False)
    val_ids[["SK_ID_CURR"]].to_csv(val_path, index=False)

    print("✅ Splits creados:")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")

if __name__ == "__main__":
    main()
