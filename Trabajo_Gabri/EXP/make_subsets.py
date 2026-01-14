import pandas as pd

# Cargar dataset full
df = pd.read_parquet("data/processed/features.parquet")

# Mezclar para que sea representativo
df = df.sample(frac=1, random_state=42)

# Subsets
df_1k = df.head(1_000)
df_10k = df.head(10_000)
df_50k = df.head(50_000)

# Guardar
df_1k.to_parquet("data/processed/features_1k.parquet", index=False)
df_10k.to_parquet("data/processed/features_10k.parquet", index=False)
df_50k.to_parquet("data/processed/features_50k.parquet", index=False)

print("Subsets creados:")
print("1k :", df_1k.shape)
print("10k:", df_10k.shape)
print("50k:", df_50k.shape)