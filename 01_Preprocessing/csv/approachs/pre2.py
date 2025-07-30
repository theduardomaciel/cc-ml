"""
APPROACH 2: Preenchimento de dados faltantes com valores aleatórios realistas

O script lê um arquivo CSV, substitui valores 0 por NaN em colunas específicas,
remove linhas com NaN em colunas críticas, e preenche os NaN restantes com valores aleatórios
gerados a partir da média e desvio padrão das colunas, garantindo que os valores sejam biologicamente válidos.
"""

import pandas as pd
import numpy as np

# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "../versions/diabetes_dataset_clean2.csv"

# Lê o CSV
df = pd.read_csv(input_csv)

# Substitui 0 por NaN em colunas onde 0 não é fisiologicamente válido
zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_nan] = df[zero_as_nan].replace(0, pd.NA)

# Remove linhas com valores faltantes em colunas com poucos NaNs
columns_to_remove_rows = ["Glucose", "BloodPressure", "BMI"]
df = df.dropna(subset=columns_to_remove_rows)


# Gera valores aleatórios realistas para colunas com muitos dados faltantes
def fill_with_realistic_random(df, column):
    mean = df[column].dropna().mean()
    std = df[column].dropna().std()
    n_missing = df[column].isna().sum()

    # Gera valores aleatórios com distribuição semelhante
    random_values = np.random.normal(loc=mean, scale=std, size=n_missing)

    # Garante que os valores sejam positivos (biologicamente válidos)
    random_values = np.clip(random_values, a_min=0, a_max=None)

    df.loc[df[column].isna(), column] = random_values


# Aplica para colunas com muitos NaNs
fill_with_realistic_random(df, "SkinThickness")
fill_with_realistic_random(df, "Insulin")

# Salva o novo CSV
df.to_csv(output_csv, index=False)

print(f"Arquivo limpo e preenchido salvo como: {output_csv}")
