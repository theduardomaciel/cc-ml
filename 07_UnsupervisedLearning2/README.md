# Atividade 4 - Descoberta dos perfis de olhos

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">
  <img alt="Atividade 04 - Clusterização" src="/.github/cover_light.png">
</picture>

## 📋 Sobre a atividade

Análise de **aprendizado não-supervisionado** para descobrir perfis/padrões de espessura epitelial em mapas oculares. O projeto compara três algoritmos de clustering (**K-Means**, **DBSCAN** e **K-Medoids**) para identificar grupos naturais de olhos com características similares de espessura do epitélio.

### 🎯 Objetivo

Descobrir **quais são os perfis de espessura epitelial** que existem na base de dados, agrupando olhos com características similares e comparando diferentes técnicas de clusterização.

### 📊 Descrição dos Dados

**Variáveis de Identificação:**
- `Index` = índice
- `pID` = ID do paciente
- `Age` = idade
- `Gender` = gênero/sexo
- `Eye` = olho (OS=esquerdo; OD=direito)

**Variáveis de Espessura Epitelial (μm):**
- `C` = central
- `S` = superior
- `ST` = superior temporal
- `T` = temporal
- `IT` = inferior temporal
- `I` = inferior
- `IN` = inferior nasal
- `N` = nasal
- `SN` = superior nasal

### 🧭 Distribuição Espacial (Rosa dos Ventos)

A disposição espacial das regiões no olho pode ser visualizada como:

|        |       |        |
|--------|-------|--------|
| **ST** | **S** | **SN** |
| **T**  | **C** | **N**  |
| **IT** | **I** | **IN** |

> 💡 Considere o enantiomorfismo dos olhos ao interpretar os resultados!

## 🚀 Como Executar

### 1️⃣ Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Execução do Projeto

**Script Principal:**

```bash
python clustering.py
```

O script executa automaticamente:
- ✅ **Pré-processamento** com validação de idade e tratamento de valores ausentes
- ✅ **Remoção de outliers** usando método IQR
- ✅ **Otimização automática** dos parâmetros de cada algoritmo
- ✅ **Clustering comparativo** com K-Means, DBSCAN e K-Medoids
- ✅ **Geração de visualizações** comparativas
- ✅ **Exportação de resultados** em CSV

**Execução com Outliers:**

O script executa duas análises:
1. **COM remoção de outliers** → `results_with_outlier_removal/`
2. **SEM remoção de outliers** → `results_without_outlier_removal/`

**Importar Módulos (Para Scripts Customizados):**

```python
from preprocessing import load_and_preprocess
from clustering import run_clustering, optimize_kmeans, optimize_dbscan

# Carregar e preprocessar dados
df, scaled_data, features, scaler = load_and_preprocess(
    'data/RTVue_20221110_MLClass.csv',
    remove_outliers=True
)

# Executar clustering
results, df, scaled_data, features = run_clustering(
    'data/RTVue_20221110_MLClass.csv',
    output_dir='results',
    remove_outliers=True
)
```

## 📁 Estrutura do Projeto

```
07_UnsupervisedLearning2/
├── data/
│   ├── RTVue_20221110_MLClass.csv         # Dataset de espessura epitelial
│   └── RTVue_20221110_MLClass.xlsx        # Versão Excel
├── results_with_outlier_removal/          # Resultados COM remoção de outliers
│   ├── clustering_results.csv             # Métricas comparativas
│   └── clustering_comparison.png          # Visualização comparativa
├── results_without_outlier_removal/       # Resultados SEM remoção de outliers
│   ├── clustering_results.csv             # Métricas comparativas
│   └── clustering_comparison.png          # Visualização comparativa
├── preprocessing.py                       # Módulo: Pré-processamento de dados
├── clustering.py                          # Módulo: Algoritmos de clustering
├── requirements.txt
└── README.md
```

### 🎯 Módulos Principais

- **`preprocessing.py`**: Carregamento, limpeza e normalização dos dados
  - Validação de idade
  - Remoção de valores ausentes
  - Detecção e remoção de outliers (IQR ou Z-Score)
  - Normalização com StandardScaler

- **`clustering.py`**: Implementação dos algoritmos de clustering
  - K-Means com otimização automática de k
  - DBSCAN com otimização de eps e min_samples
  - K-Medoids para comparação
  - Geração de visualizações comparativas

## 🔬 Metodologia

### 1. Pré-processamento (`preprocessing.py`)
- Carregamento dos dados do arquivo CSV
- Validação e limpeza de idade (0-120 anos)
- Seleção das 9 variáveis de espessura epitelial
- Remoção de valores ausentes
- **Detecção e remoção de outliers:**
  - **Método IQR** (Interquartile Range): multiplier = 1.5
  - **Método Z-Score**: threshold = 3
- Normalização dos dados (StandardScaler)

### 2. Algoritmos de Clustering

#### **K-Means**
- Otimização automática do número de clusters (k=2 a k=10)
- Métricas: Silhouette Score e Inércia
- K fixo em 3 para consistência nas comparações
- Parâmetros: `n_init=10`, `random_state=42`

#### **DBSCAN**
- Otimização automática de `eps` usando k-NN (k=5)
- Teste de múltiplos valores de `eps` e `min_samples`
- Detecção automática de ruído (outliers)
- Ideal para clusters de formas irregulares

#### **K-Medoids**
- Alternativa robusta ao K-Means
- Usa medóides (amostras reais) em vez de centróides
- Mesma quantidade de clusters do K-Means (k=3)
- Mais resistente a outliers

### 3. Avaliação e Comparação
- **Silhouette Score**: Coesão vs separação dos clusters
- **Calinski-Harabasz Score**: Razão de dispersão entre/dentro clusters
- **Davies-Bouldin Score**: Similaridade entre clusters
- Distribuição de amostras por cluster
- Visualizações comparativas (boxplots e distribuições)

### 4. Geração de Resultados
- Duas execuções completas (com/sem outliers)
- CSV com métricas comparativas
- Visualizações comparativas dos três algoritmos
- Análise da distribuição das features por cluster

## 📊 Resultados Esperados

O projeto gera duas análises completas:

### 📁 `results_with_outlier_removal/`
- **`clustering_results.csv`**: Tabela com métricas dos três algoritmos
- **`clustering_comparison.png`**: Visualização comparativa com:
  - Distribuição de amostras por cluster
  - Boxplots das features normalizadas
  - Percentuais de cada cluster

### 📁 `results_without_outlier_removal/`
- Mesma estrutura, mas sem remoção de outliers
- Útil para comparar o impacto da limpeza de dados

### 📈 Informações Incluídas

✅ **Comparação de Algoritmos**: K-Means vs DBSCAN vs K-Medoids  
✅ **Métricas de Qualidade**: Silhouette, Calinski-Harabasz, Davies-Bouldin  
✅ **Distribuição de Clusters**: Quantidade e percentual de amostras  
✅ **Detecção de Ruído**: Identificação de outliers pelo DBSCAN  
✅ **Visualizações Comparativas**: Gráficos profissionais para análise

## 📈 Métricas de Avaliação

- **Silhouette Score** (-1 a 1): Mede quão similar um ponto é ao seu cluster vs outros clusters. **Maior é melhor.**
- **Calinski-Harabasz Score**: Razão da dispersão entre clusters e dentro dos clusters. **Maior é melhor.**
- **Davies-Bouldin Score**: Média da similaridade entre cada cluster e seu mais similar. **Menor é melhor.**
- **Inércia** (K-Means): Soma das distâncias quadradas dos pontos aos centróides. **Menor é melhor.**
- **Número de Ruído** (DBSCAN): Quantidade de pontos identificados como outliers.

## 🔧 Personalização

### Ajustar Número de Clusters

Edite em `clustering.py`:
```python
best_k = 3  # Altere o valor conforme necessário
```

### Ajustar Método de Remoção de Outliers

Em `preprocessing.py`, na função `load_and_preprocess`:
```python
# Opção 1: IQR (padrão)
outlier_method="iqr"
iqr_multiplier=1.5  # Ajuste a sensibilidade

# Opção 2: Z-Score
outlier_method="zscore"
zscore_threshold=3  # Ajuste o limiar
```

### Desabilitar Remoção de Outliers

Em `clustering.py`:
```python
results = run_clustering(
    data_path,
    output_dir="results",
    remove_outliers=False  # Desabilita remoção
)
```

## 📚 Dependências

- pandas >= 2.1.3
- numpy >= 1.26.2
- scikit-learn >= 1.3.2
- scikit-learn-extra >= 0.3.0 (K-Medoids)
- scipy >= 1.11.4 (detecção de outliers)
- matplotlib >= 3.8.2
- seaborn >= 0.13.0

## 🎓 Conceitos Aplicados

- **Aprendizado Não-Supervisionado**: Descoberta de padrões sem labels
- **Normalização**: StandardScaler para equalizar escalas
- **Detecção de Outliers**: IQR e Z-Score
- **Otimização de Hiperparâmetros**: Grid search para DBSCAN, Elbow Method para K-Means
- **Avaliação de Clusters**: Múltiplas métricas de validação interna
- **Análise Comparativa**: Avaliação de diferentes algoritmos no mesmo dataset
