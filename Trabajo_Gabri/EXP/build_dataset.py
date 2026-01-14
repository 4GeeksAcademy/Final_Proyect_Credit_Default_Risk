from src.features.load_data import load_data

# Cargar datos
data = load_data()

app = data["app"]
prev_agg = data["prev_agg"]
bureau_agg = data["bureau_agg"]
pos_agg = data["pos_agg"]
inst_agg = data["inst_agg"]


# Merge final a nivel cliente (1 fila = 1 cliente)
X = (
    app
    .merge(prev_agg, on="SK_ID_CURR", how="left")
    .merge(bureau_agg, on="SK_ID_CURR", how="left")
    .merge(pos_agg, on="SK_ID_CURR", how="left")
    .merge(inst_agg, on="SK_ID_CURR", how="left")
)

# Guardar dataset final
X.to_parquet("data/processed/features.parquet", index=False)

print("Dataset final creado")
print("Shape:", X.shape)
