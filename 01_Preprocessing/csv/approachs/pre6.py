"""
APPROACH 6:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Caminho para o arquivo CSV de entrada
input_csv = "../diabetes_dataset.csv"

# Caminho para o arquivo CSV de saída
output_csv = "../versions/clean7.csv"

# Lê o CSV
df = pd.read_csv(input_csv)

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
df = df[features]

# Separa o dataset em dois dataframes, um para cada classe de "Outcome"
dataset = df[df["Outcome"] == 0]
dataset1 = df[df["Outcome"] == 1]

# Calcula as medianas de cada dataframe
median = dataset.median()
median1 = dataset1.median()

# Preenche os valores ausentes com as medianas
dataset = dataset.fillna(median)
dataset1 = dataset1.fillna(median1)
merge_tables = pd.concat([dataset, dataset1])

# Normaliza os dados
outcome_column = pd.DataFrame(merge_tables["Outcome"])
merge_tables_normalized = merge_tables.drop(columns=["Outcome"])

# Normaliza os dados usando a normalização L2
scaler = preprocessing.normalize(merge_tables_normalized, axis=0)
merge_tables_normalized = pd.DataFrame(scaler, columns=merge_tables_normalized.columns)

# Adiciona a coluna "Outcome" de volta ao dataframe normalizado
merge_tables_normalized.insert(
    len(merge_tables_normalized.columns), "Outcome", outcome_column
)

# Salva o dataframe normalizado em um novo arquivo CSV
merge_tables_normalized.to_csv(output_csv, index=False)

print(f"Dataset limpa e salva em {output_csv}")
