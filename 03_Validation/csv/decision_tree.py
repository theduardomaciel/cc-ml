"""
Atividade 03 - Avaliação de classificadores (com Árvore de Decisão)

Validação intensiva de k para k-NN no dataset Abalone, com pré-processamento
por Pipeline (One-Hot para categóricos, padronização para numéricos),
validação cruzada repetida e envio opcional do melhor modelo ao servidor.
"""

import argparse  # Para leitura de argumentos de linha de comando
import json  # Para manipulação de arquivos JSON
import os  # Para acessar variáveis de ambiente e sistema
from pathlib import Path  # Para manipulação de caminhos de arquivos

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# Caminhos padrão (relativos a esta pasta)
DATASET_PATH = Path(__file__).parent / "abalone_dataset.csv"
APP_PATH = Path(__file__).parent / "abalone_app.csv"
OUT_DIR = Path(__file__).parent / "out" / "tree"
OUT_DIR.mkdir(exist_ok=True)

SERVER_URL = "https://aydanomachado.com/mlclass/03_Validation.php"


# Função para identificar colunas alvo, numéricas e categóricas
def infer_columns(df: pd.DataFrame, target_hint: str = "type"):
    # Define a coluna alvo (target) e separa as features
    if target_hint in df.columns:
        target_col = target_hint
    else:
        target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    # Identifica colunas categóricas e numéricas
    cat_cols = [
        c
        for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return target_col, feature_cols, num_cols, cat_cols


# Função que constrói o pipeline de pré-processamento e modelo
def build_pipeline(num_cols, cat_cols):
    # Pré-processamento: numéricos recebem imputação e padronização, categóricos recebem imputação e one-hot
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    # Modelo: DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    # Pipeline completo: pré-processamento + modelo
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    return pipe


# Função para converter previsões em lista serializável
def to_list_of_py(obj):
    if isinstance(obj, tuple):
        obj = obj[0]
    arr = np.asarray(obj).ravel()
    if arr.dtype.kind in {"i", "u"}:
        return [int(x) for x in arr]
    return [str(x) for x in arr]


# Função principal do script
def main():
    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Validação de Decision Tree no Abalone (CSV)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Caminho do abalone_dataset.csv",
    )
    parser.add_argument(
        "--app", type=str, default=str(APP_PATH), help="Caminho do abalone_app.csv"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=10, help="Número de folds na validação cruzada"
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=3,
        help="Número de repetições da validação cruzada",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed para reprodutibilidade"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Número de jobs paralelos (-1 usa todos). No Windows, use 1 se tiver erro de multiprocessing.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Se definido, envia as previsões do melhor modelo ao servidor",
    )
    parser.add_argument(
        "--dev-key",
        type=str,
        default=os.environ.get("DEV_KEY", "COLOCAR_SUA_KEY_AQUI"),
        help="Chave do desenvolvedor (pode usar variável de ambiente DEV_KEY)",
    )
    args = parser.parse_args()

    # Carrega o dataset de treino
    df = pd.read_csv(args.dataset)
    target_col, feature_cols, num_cols, cat_cols = infer_columns(df)
    X = df[feature_cols]
    y = df[target_col]
    print(f"Coluna alvo: {target_col}")
    print(f"Features numéricas: {num_cols}")
    print(f"Features categóricas: {cat_cols}")

    # Cria pipeline de pré-processamento e modelo
    pipe = build_pipeline(num_cols, cat_cols)

    # Validação cruzada repetida para avaliação robusta
    rskf = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.random_state,
    )

    # Define número de jobs para paralelismo
    n_jobs = (
        -1
        if args.n_jobs is None and os.name != "nt"
        else (1 if args.n_jobs is None else args.n_jobs)
    )

    # Espaço de busca de hiperparâmetros para Decision Tree
    param_grid = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    # Busca em grade (GridSearchCV) para encontrar melhores hiperparâmetros
    search = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=rskf,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    print("Iniciando busca por Decision Tree...")
    search.fit(X, y)
    best_score = search.best_score_
    best_params = search.best_params_
    print("Melhor acurácia (CV):", best_score)
    print("Melhores parâmetros:", best_params)
    best_model = search.best_estimator_

    # Carrega dados de aplicação (teste)
    df_app = pd.read_csv(args.app)
    # Garante que só as colunas de features presentes no treino sejam usadas
    X_app = df_app[[c for c in feature_cols if c in df_app.columns]]
    # Faz as previsões
    y_pred = best_model.predict(X_app)

    # Salva previsões em CSV e JSON
    pred_csv = OUT_DIR / "predictions.csv"
    pd.DataFrame({target_col: y_pred}).to_csv(pred_csv, index=False)
    pred_json = OUT_DIR / "predictions.json"
    pred_json.write_text(json.dumps(to_list_of_py(y_pred)))
    print(f"Previsões salvas em: {pred_csv} e {pred_json}")

    # Envia resultados ao servidor, se solicitado
    if args.send:
        dev_key = args.dev_key
        if not dev_key or dev_key == "nebulosa":
            print(
                "DEV_KEY não informado. Use --dev-key ou defina a variável de ambiente DEV_KEY."
            )
        else:
            payload = {
                "dev_key": dev_key,
                "predictions": json.dumps(to_list_of_py(y_pred)),
            }
            try:
                print("Enviando resultados ao servidor...")
                r = requests.post(url=SERVER_URL, data=payload, timeout=60)
                print("Resposta do servidor:\n", r.text)
                (OUT_DIR / "response.txt").write_text(r.text)
                print(f"Resposta salva em: {OUT_DIR / 'response.txt'}")
            except Exception as e:
                print("Falha ao enviar resultados:", e)


# Execução do script principal
if __name__ == "__main__":
    main()
