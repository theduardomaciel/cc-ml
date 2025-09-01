"""
Atividade 03 - Avaliação de classificadores (com k-NN)

Validação intensiva de k para k-NN no dataset Abalone, com pré-processamento
por Pipeline (One-Hot para categóricos, padronização para numéricos),
validação cruzada repetida e envio opcional do melhor modelo ao servidor.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Caminhos padrão (relativos a esta pasta)
DATASET_PATH = Path(__file__).parent / "abalone_dataset.csv"
APP_PATH = Path(__file__).parent / "abalone_app.csv"
OUT_DIR = Path(__file__).parent / "out" / "knn"
OUT_DIR.mkdir(exist_ok=True)

SERVER_URL = "https://aydanomachado.com/mlclass/03_Validation.php"


def infer_columns(df: pd.DataFrame, target_hint: str = "Type"):
    # Define a coluna alvo
    if target_hint in df.columns:
        target_col = target_hint
    else:
        target_col = df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]

    # Detecta tipos de coluna dinamicamente
    cat_cols = [
        c
        for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return target_col, feature_cols, num_cols, cat_cols


def build_pipeline(num_cols, cat_cols):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = KNeighborsClassifier()

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def grid_for_k(min_k: int, max_k: int, step: int):
    ks = list(range(min_k, max_k + 1, step))
    return {
        "model__n_neighbors": ks,
        "model__weights": ["uniform", "distance"],
        "model__metric": ["minkowski"],
        "model__p": [1, 2],  # Manhattan e Euclidiana
    }


def to_list_of_py(obj):
    """Converte saída de predição (array/tuple) em lista serializável simples.

    - Se vier tupla (ex.: alguns métodos retornam (pred, _)), usa o primeiro item.
    - Achata para 1-D.
    - Converte ints para int nativo, demais para str.
    """
    if isinstance(obj, tuple):
        obj = obj[0]
    arr = np.asarray(obj).ravel()
    if arr.dtype.kind in {"i", "u"}:  # inteiro assinado/sem sinal
        return [int(x) for x in arr]
    return [str(x) for x in arr]


def main():
    parser = argparse.ArgumentParser(description="Validação de k-NN no Abalone (CSV)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Caminho do abalone_dataset.csv",
    )
    parser.add_argument(
        "--app", type=str, default=str(APP_PATH), help="Caminho do abalone_app.csv"
    )
    parser.add_argument("--min-k", type=int, default=1, help="k mínimo")
    parser.add_argument("--max-k", type=int, default=75, help="k máximo")
    parser.add_argument("--step", type=int, default=2, help="passo para k")
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
        "--use-best",
        action="store_true",
        help="Usa configuração salva em best_params.json e pula a busca",
    )
    parser.add_argument(
        "--best-file",
        type=str,
        default=str(OUT_DIR / "best_params.json"),
        help="Caminho do arquivo JSON com melhor configuração",
    )
    parser.add_argument(
        "--dev-key",
        type=str,
        default=os.environ.get("DEV_KEY", "COLOCAR_SUA_KEY_AQUI"),
        help="Chave do desenvolvedor (pode usar variável de ambiente DEV_KEY)",
    )
    args = parser.parse_args()

    # Carrega dados
    df = pd.read_csv(args.dataset)
    target_col, feature_cols, num_cols, cat_cols = infer_columns(df)

    X = df[feature_cols]
    y = df[target_col]

    print(f"Coluna alvo: {target_col}")
    print(f"Features numéricas: {num_cols}")
    print(f"Features categóricas: {cat_cols}")

    # Pipeline
    pipe = build_pipeline(num_cols, cat_cols)

    best_file = Path(args.best_file)

    if args.use_best and best_file.exists():
        # Carrega melhor configuração anterior e reusa
        print(f"Carregando melhor configuração de: {best_file}")
        with open(best_file, "r", encoding="utf-8") as f:
            best_payload = json.load(f)
        best_params = best_payload.get("best_params", {})
        # Aplica e ajusta em todos os dados
        pipe.set_params(**best_params)
        best_model = pipe.fit(X, y)
        best_score = best_payload.get("best_score")
        print("Usando best_params carregado:", best_params)
        if best_score is not None:
            print("Acurácia (CV) anterior:", best_score)
    else:
        # Grid de parâmetros
        param_grid = grid_for_k(args.min_k, args.max_k, args.step)

        # Validação cruzada repetida
        rskf = RepeatedStratifiedKFold(
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
            random_state=args.random_state,
        )

        # Define n_jobs com fallback seguro no Windows
        if args.n_jobs is None:
            n_jobs = -1 if os.name != "nt" else 1
        else:
            n_jobs = args.n_jobs

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=rskf,
            n_jobs=n_jobs,
            verbose=1,
            refit=True,  # mantém o melhor modelo ajustado no final
            return_train_score=False,
        )

        print("Iniciando busca por k...")
        search.fit(X, y)

        best_score = search.best_score_
        best_params = search.best_params_
        print("Melhor acurácia (CV):", best_score)
        print("Melhores parâmetros:", best_params)

        # Salva resultados da busca
        results_df = pd.DataFrame(search.cv_results_)
        results_path = OUT_DIR / "knn_grid_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Resultados completos salvos em: {results_path}")

        # Salva best_params.json com metadados úteis
        best_payload = {
            "best_params": best_params,
            "best_score": best_score,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "cv": {
                "splits": args.cv_splits,
                "repeats": args.cv_repeats,
                "random_state": args.random_state,
            },
            "search_space": {
                "min_k": args.min_k,
                "max_k": args.max_k,
                "step": args.step,
            },
        }
        with open(best_file, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, ensure_ascii=False, indent=2)
        print(f"Melhor configuração salva em: {best_file}")

        # Melhor modelo ajustado pela busca
        best_model = search.best_estimator_
    df_app = pd.read_csv(args.app)
    # Garantir mesmas colunas de features (colunas extras serão ignoradas no transformer)
    X_app = df_app[[c for c in feature_cols if c in df_app.columns]]
    y_pred = best_model.predict(X_app)

    # Saídas locais
    pred_csv = OUT_DIR / "predictions.csv"
    pd.DataFrame({target_col: y_pred}).to_csv(pred_csv, index=False)
    pred_json = OUT_DIR / "predictions.json"
    pred_json.write_text(json.dumps(to_list_of_py(y_pred)))
    print(f"Previsões salvas em: {pred_csv} e {pred_json}")

    # Envio opcional
    if args.send:
        dev_key = args.dev_key
        if not dev_key or dev_key == "COLOCAR_SUA_KEY_AQUI":
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


if __name__ == "__main__":
    main()
