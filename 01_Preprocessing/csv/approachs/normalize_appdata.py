data_file = "../diabetes_app.csv"
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Carregar o dataset
df = pd.read_csv(data_file)

# Normalizar apenas colunas numéricas
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Gerar nome do arquivo paralelo
base, ext = os.path.splitext(data_file)
normalized_file = base + "_normalized" + ext

# Salvar o novo arquivo
df.to_csv(normalized_file, index=False)
print(f"Arquivo normalizado salvo em: {normalized_file}")
