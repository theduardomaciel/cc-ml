"""
APPROACH 3.1: Preenchimento de dados faltantes com base na mediana

O script lê um arquivo CSV, substitui valores 0 por NaN em colunas específicas,
e preenche os valores ausentes com a mediana de cada coluna, garantindo que os valores
sejam biologicamente válidos.
"""

import pandas as pd

# Caminho para o arquivo CSV de entrada
entrada_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
saida_csv = "../versions/clean3_1.csv"

# Lê o CSV
df = pd.read_csv(entrada_csv)

# Substitui 0 por NaN onde 0 não é fisiologicamente válido
zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_nan] = df[zero_as_nan].replace(0, pd.NA)

# Preenche os valores ausentes com a mediana de cada coluna
df_preenchido = df.fillna(
    df.median(numeric_only=True)
)  # Substituir por df.mean() para fazer a média

# Salva o novo CSV
df_preenchido.to_csv(saida_csv, index=False)

print("Valores ausentes preenchidos com a mediana de cada coluna.")
print(f"Arquivo limpo salvo como: {saida_csv}")
