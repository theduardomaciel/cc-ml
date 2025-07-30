import pandas as pd
from sklearn.preprocessing import StandardScaler

# Caminho para o arquivo CSV de entrada
entrada_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
saida_csv = "../versions/clean4.csv"

# Lê o CSV
df = pd.read_csv(entrada_csv)

# Substitui 0 por NaN em colunas onde 0 não é fisiologicamente válido
zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_nan] = df[zero_as_nan].replace(0, pd.NA)

# Preenche valores ausentes com a mediana de cada coluna
df_preenchido = df.fillna(df.median(numeric_only=True))

# Separa X (features) e y (alvo)
X = df_preenchido.drop(columns=["Outcome"])
y = df_preenchido["Outcome"]

# Normaliza as features com StandardScaler
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Reconstrói o DataFrame com colunas e adiciona a coluna "Outcome"
df_normalizado = pd.DataFrame(X_normalizado, columns=X.columns)
df_normalizado["Outcome"] = y.values

# Salva o novo CSV
df_normalizado.to_csv(saida_csv, index=False)

print(
    "Valores ausentes preenchidos com a mediana e dados normalizados com StandardScaler."
)
print(f"Arquivo final salvo como: {saida_csv}")
