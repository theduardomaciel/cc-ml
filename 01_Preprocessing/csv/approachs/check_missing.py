import pandas as pd

# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "diabetes_dataset1.csv"

# Lê o CSV
df = pd.read_csv(input_csv)

# Lista quantas linhas estão faltantes em cada coluna
missing_rows = df.isnull().sum()
print("Linhas faltantes por coluna:")
print(missing_rows)

""" # Remove as linhas com qualquer valor faltante
clean_df = df.dropna()

# Salva o novo CSV
clean_df.to_csv(output_csv, index=False)

print(f"{len(df) - len(clean_df)} linhas com dados faltantes foram removidas.")
print(f"Arquivo limpo salvo como: {output_csv}")
 """
