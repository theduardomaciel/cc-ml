# Como importar e executar os scripts individualmente

Para facilitar a execução dos scripts de forma independente, é possível importar os módulos do diretório atual diretamente:

```python
from preprocessing import load_and_preprocess
from clustering import run_clustering, optimize_kmeans, optimize_dbscan
from demographic_analysis import analyze_demographics

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

# Análise demográfica
df_analyzed, age_results, gender_results, eye_results = analyze_demographics(
    'data/RTVue_20221110_MLClass.csv',
    n_clusters=3,
    output_dir='results_demographic'
)
```