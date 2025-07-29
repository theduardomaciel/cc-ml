import pandas as pd

# Caminho para o arquivo CSV de entrada
entrada_csv = "diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
saida_csv = "diabetes_dataset_clean1.csv"

# Lê o CSV
df = pd.read_csv(entrada_csv)

# Remove as linhas com qualquer valor faltante
df_limpo = df.dropna()

# Salva o novo CSV
df_limpo.to_csv(saida_csv, index=False)

print(f"{len(df) - len(df_limpo)} linhas com dados faltantes foram removidas.")
print(f"Arquivo limpo salvo como: {saida_csv}")
