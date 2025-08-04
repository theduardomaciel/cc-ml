import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("diabetes_dataset.csv")

# Ajustes iniciais
sns.set(style="whitegrid")

# 1. Histogramas das variáveis numéricas
df.hist(figsize=(12, 10), bins=20, edgecolor="black")
plt.suptitle("Distribuição das variáveis", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Boxplots por resultado
df_melted = df.melt(id_vars="Outcome", var_name="Variavel", value_name="Valor")
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_melted, x="Variavel", y="Valor", hue="Outcome")
plt.title("Distribuição das variáveis por resultado (Outcome)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# 4. Gráfico de dispersão entre variáveis selecionadas e Outcome
features = ["Glucose", "BMI", "Age", "Insulin"]
plt.figure(figsize=(12, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=df, x=col, hue="Outcome", kde=True, element="step")
    plt.title(f"{col} vs Outcome")
plt.tight_layout()
plt.show()
