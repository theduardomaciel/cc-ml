"""
APPROACH 1: Remoção de linhas com dados faltantes

O script lê um arquivo CSV, remove linhas com qualquer valor faltante e salva o resultado em um novo arquivo CSV.
"""

import pandas as pd


# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "../versions/clean1.csv"

# Lê o CSV
df = pd.read_csv(input_csv)

# Remove as linhas com qualquer valor faltante
clean_df = df.dropna()

# Salva o novo CSV
clean_df.to_csv(output_csv, index=False)

print(f"{len(df) - len(clean_df)} linhas com dados faltantes foram removidas.")
print(f"Arquivo limpo salvo como: {output_csv}")
