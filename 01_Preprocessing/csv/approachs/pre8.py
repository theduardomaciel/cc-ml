"""
APPROACH:
1. Remoção de linhas com >2 dados faltantes.
2. Imputação iterativa para os valores restantes.
3. Escalonamento com StandardScaler.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "../versions/clean8.csv"

# Lê o CSV
df = pd.read_csv(input_csv)
initial_rows = len(df)

# --- 2. Descarta linhas com >2 valores ausentes ---
# Mantém linhas que têm no máximo 2 valores ausentes (ou seja, pelo menos 7 valores não nulos)
df_thresh = df.dropna(thresh=df.shape[1] - 2)
rows_after_drop = len(df_thresh)
print(
    f"{initial_rows - rows_after_drop} linhas com mais de 2 valores ausentes foram removidas."
)

# --- 3. Imputação iterativa dos valores faltantes restantes ---
# A imputação é feita em todas as colunas, pois todas são numéricas
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df_thresh), columns=df_thresh.columns)
print("Imputação iterativa concluída.")

# Separa features (X) e target (y)
X = df_imputed.drop("Outcome", axis=1)
y = df_imputed["Outcome"]

# --- 5. Escalonamento com StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cria um novo DataFrame com as features escalonadas
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Junta as features escalonadas com a coluna de target
final_df = pd.concat([df_scaled, y.reset_index(drop=True)], axis=1)
print("Escalonamento com StandardScaler concluído.")

# Salva o novo CSV
final_df.to_csv(output_csv, index=False)

print(f"Arquivo processado salvo como: {output_csv}")
