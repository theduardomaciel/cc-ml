"""
Atividade 03 - Comparação de múltiplos classificadores no Abalone

- Pipeline genérico com pré-processamento
- Grid search para múltiplos modelos
- Validação cruzada repetida
- Salva resultados detalhados e tabela comparativa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


DATASET_PATH = Path(__file__).parent / "abalone_dataset.csv"
APP_PATH = Path(__file__).parent / "abalone_app.csv"
OUT_DIR = Path(__file__).parent / "out" / "all_models"
OUT_DIR.mkdir(exist_ok=True)
SERVER_URL = "https://aydanomachado.com/mlclass/03_Validation.php"


def infer_columns(df: pd.DataFrame, target_hint="Type"):
    """
    Infere e retorna as colunas alvo, de características, numéricas e categóricas de um DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada contendo os dados.
        target_hint (str, opcional): Nome sugerido para a coluna alvo. Padrão é "Type".

    Retorna:
        tuple:
            - target_col (str): Nome da coluna alvo.
            - feature_cols (list): Lista com os nomes das colunas de características (exceto a alvo).
            - num_cols (list): Lista com os nomes das colunas numéricas.
            - cat_cols (list): Lista com os nomes das colunas categóricas.
    """
    if target_hint in df.columns:
        target_col = target_hint
    else:
        target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [
        c
        for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return target_col, feature_cols, num_cols, cat_cols


def build_pipeline(model, num_cols, cat_cols):
    """
    Cria um pipeline de pré-processamento e modelagem para dados tabulares.

    Parâmetros:
        model: objeto
            O estimador ou modelo de machine learning a ser utilizado no pipeline.
        num_cols: lista de str
            Lista com os nomes das colunas numéricas do conjunto de dados.
        cat_cols: lista de str
            Lista com os nomes das colunas categóricas do conjunto de dados.

    Retorna:
        pipe: sklearn.pipeline.Pipeline
            Um pipeline contendo o pré-processamento das variáveis numéricas e categóricas,
            seguido pelo modelo especificado.
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def get_model_grid(model_name):
    """
    Retorna o estimador e o grid de hiper-parâmetros para busca em grade, de acordo com o nome do modelo especificado.

    Parâmetros:
        model_name (str): Nome do modelo desejado. Opções válidas incluem:
            - "knn"
            - "decision_tree"
            - "random_forest"
            - "gradient_boosting"
            - "svm"
            - "mlp"
            - "naive_bayes"
            - "logistic_regression"

    Retorna:
        tuple:
            - Um estimador sklearn correspondente ao modelo solicitado.
            - Um dicionário contendo os hiper-parâmetros e seus valores para busca em grade.

    Exceções:
        ValueError: Se o nome do modelo fornecido não for reconhecido.
    """
    if model_name == "knn":
        return KNeighborsClassifier(), {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=42), {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5, 10],
        }
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=42), {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        }
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42), {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        }
    elif model_name == "svm":
        return SVC(probability=True, random_state=42), {
            "model__C": [0.1, 1],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    elif model_name == "mlp":
        return MLPClassifier(max_iter=1000, random_state=42), {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001],
        }
    elif model_name == "naive_bayes":
        return GaussianNB(), {}
    elif model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42
        ), {"model__C": [0.1, 1, 10]}
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")


def to_list_of_py(obj):
    """
    Converte um objeto em uma lista de valores Python.

    Se o objeto for uma tupla, utiliza apenas o primeiro elemento.
    Em seguida, converte o objeto para um array NumPy unidimensional.
    Se o tipo dos dados for inteiro (signed ou unsigned), retorna uma lista de inteiros.
    Caso contrário, retorna uma lista de strings.

    Parâmetros:
        obj: Objeto a ser convertido (pode ser tupla, lista, array, etc).

    Retorna:
        list: Lista de inteiros ou strings, dependendo do tipo dos dados do objeto.
    """
    if isinstance(obj, tuple):
        obj = obj[0]
    arr = np.asarray(obj).ravel()
    if arr.dtype.kind in {"i", "u"}:
        return [int(x) for x in arr]
    return [str(x) for x in arr]


def main():
    parser = argparse.ArgumentParser(
        description="Comparação de múltiplos classificadores no Abalone"
    )
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--app", type=str, default=str(APP_PATH))
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--dev-key", type=str, default="COLOCAR_SUA_KEY_AQUI")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    target_col, feature_cols, num_cols, cat_cols = infer_columns(df)
    X = df[feature_cols]
    y = df[target_col]

    models = [
        "knn",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "mlp",
        "naive_bayes",
        "logistic_regression",
    ]

    summary = []

    for m in models:
        print(f"\n===== Treinando {m} =====")
        model, param_grid = get_model_grid(m)
        pipe = build_pipeline(model, num_cols, cat_cols)

        rskf = RepeatedStratifiedKFold(
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=rskf,
            n_jobs=args.n_jobs,
            verbose=1,
            refit=True,
        )
        search.fit(X, y)

        print(f"Melhor acurácia CV para {m}: {search.best_score_:.4f}")
        print("Melhores parâmetros:", search.best_params_)

        # Salva resultados detalhados
        pd.DataFrame(search.cv_results_).to_csv(
            OUT_DIR / f"{m}_grid_results.csv", index=False
        )

        summary.append(
            {
                "model": m,
                "best_score": search.best_score_,
                "best_params": search.best_params_,
            }
        )

        # Tabela resumo
    summary_df = pd.DataFrame(summary).sort_values("best_score", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary_models.csv", index=False)
    print("\nTabela comparativa de acurácias salva em summary_models.csv")
    print(summary_df)

    # Melhor modelo
    best_model_name = summary_df.iloc[0]["model"]
    best_params = summary_df.iloc[0]["best_params"]
    best_score = summary_df.iloc[0]["best_score"]

    # Salvar best_model.json
    best_payload = {
        "model": best_model_name,
        "best_params": best_params,
        "best_score": best_score,
    }
    (OUT_DIR / "best_model.json").write_text(json.dumps(best_payload, indent=2))
    print(f"\nMelhor modelo salvo em {OUT_DIR / 'best_model.json'}")

    # Previsões no conjunto app com melhor modelo
    print(
        f"\nUsando melhor modelo ({best_model_name}) para gerar previsões no conjunto app..."
    )
    best_model, _ = get_model_grid(best_model_name)
    best_pipe = build_pipeline(best_model, num_cols, cat_cols)
    best_pipe.set_params(**best_params)
    best_pipe.fit(X, y)

    df_app = pd.read_csv(args.app)
    X_app = df_app[[c for c in feature_cols if c in df_app.columns]]
    y_pred = best_pipe.predict(X_app)

    pd.DataFrame({target_col: y_pred}).to_csv(
        OUT_DIR / f"{best_model_name}_predictions.csv", index=False
    )
    (OUT_DIR / f"{best_model_name}_predictions.json").write_text(
        json.dumps(to_list_of_py(y_pred))
    )

    print("Previsões salvas para o melhor modelo.")

    # Envio opcional
    if args.send and args.dev_key != "COLOCAR_SUA_KEY_AQUI":
        payload = {
            "dev_key": args.dev_key,
            "predictions": json.dumps(to_list_of_py(y_pred)),
        }
        try:
            r = requests.post(SERVER_URL, data=payload, timeout=60)
            print("Resposta do servidor:\n", r.text)
        except Exception as e:
            print("Falha ao enviar:", e)


if __name__ == "__main__":
    main()
