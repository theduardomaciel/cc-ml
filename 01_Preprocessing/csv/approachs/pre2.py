"""
APPROACH 2: Preenchimento de dados faltantes com valores aleatórios realistas

O script lê um arquivo CSV, substitui valores 0 por NaN em colunas específicas,
remove linhas com NaN em colunas críticas, e preenche os NaN restantes com valores aleatórios
gerados a partir da média e desvio padrão das colunas, garantindo que os valores sejam biologicamente válidos.
"""

import pandas as pd
import numpy as np

# Caminho para o arquivo CSV de entrada
entrada_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
saida_csv = "../versions/clean2.csv"

# Lê o CSV
df = pd.read_csv(entrada_csv)

# Substitui 0 por NaN em colunas onde 0 não é fisiologicamente válido
zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_nan] = df[zero_as_nan].replace(0, pd.NA)

# Remove linhas com valores faltantes em colunas com valores críticos
colunas_para_remover_linhas = ["Glucose", "BloodPressure", "BMI"]
df = df.dropna(subset=colunas_para_remover_linhas)


# Gera valores aleatórios realistas para colunas com muitos dados faltantes
def preencher_com_aleatorio_realista(df, coluna):
    media = df[coluna].dropna().mean()
    desvio = df[coluna].dropna().std()
    n_faltantes = df[coluna].isna().sum()

    # Gera valores aleatórios com distribuição semelhante
    valores_aleatorios = np.random.normal(loc=media, scale=desvio, size=n_faltantes)

    # Garante que os valores sejam positivos (biologicamente válidos)
    valores_aleatorios = np.clip(valores_aleatorios, a_min=0, a_max=None)

    df.loc[df[coluna].isna(), coluna] = valores_aleatorios


# Aplica para colunas com muitos NaNs
preencher_com_aleatorio_realista(df, "SkinThickness")
preencher_com_aleatorio_realista(df, "Insulin")

# Salva o novo CSV
df.to_csv(saida_csv, index=False)

print(f"Arquivo limpo e preenchido salvo como: {saida_csv}")
