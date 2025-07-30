"""
APPROACH 5: Preenchimento de dados faltantes com média e remoção de linhas

O script lê um arquivo CSV, substitui valores 0 por NaN em colunas específicas,
remove linhas com valores faltantes em colunas críticas, e preenche os valores ausentes
com a média de cada coluna, garantindo que os valores sejam biologicamente válidos.
"""

import pandas as pd

# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "../versions/clean5.csv"

# Lê o CSV
df = pd.read_csv(input_csv)

# Substitui 0 por NaN em colunas onde 0 não é fisiologicamente válido
zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_nan] = df[zero_as_nan].replace(0, pd.NA)

# Remove linhas com valores faltantes em colunas com poucos NaNs
columns_to_remove_rows = ["Glucose", "BloodPressure", "BMI"]
df = df.dropna(subset=columns_to_remove_rows)

# Preenche com média as lacunas vazias nas colunas "Insulin" e "SkinThickness"
df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
df["SkinThickness"].fillna(
    df["SkinThickness"].mean(),
    inplace=True,
)  # Substituir por df["SkinThickness"].median() para usar a mediana

# Salva o novo CSV
df.to_csv(output_csv, index=False)

print(f"Arquivo limpo e preenchido salvo como: {output_csv}")
