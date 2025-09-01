"""
Avaliação de classificadores (genérico) com gráficos opcionais + salvamento em PNG
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from all_models import get_model_grid, build_pipeline, infer_columns


def save_and_show(fig, filename):
    """Salva a figura em results/ e também exibe na tela."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / filename, bbox_inches="tight")
    plt.show()


def evaluate_holdout(model, X, y, test_size=0.3, random_state=42, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Avaliação Holdout ===")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))

    if plot:
        # 📊 Matriz de confusão
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap="Blues", ax=ax
        )
        ax.set_title("Matriz de Confusão (Holdout)")
        save_and_show(fig, "holdout_confusion_matrix.png")

        # 📊 Relatório de classificação em gráfico
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose().drop("accuracy", errors="ignore")

        fig, ax = plt.subplots(figsize=(8, 5))
        df_report[["precision", "recall", "f1-score"]].plot(
            kind="bar", ax=ax, legend=True
        )
        ax.set_title("Métricas por classe (Holdout)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        save_and_show(fig, "holdout_classification_report.png")


def evaluate_kfold(model, X, y, k=10, random_state=42, plot=False):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=1)

    print(f"\n=== Avaliação K-Fold ({k}) ===")
    print("Acurácias:", scores)
    print("Média:", scores.mean())
    print("Desvio padrão:", scores.std())

    if plot:
        # 📊 Histograma das acurácias
        fig, ax = plt.subplots()
        ax.hist(scores, bins=5, edgecolor="black")
        ax.set_title(f"Distribuição das acurácias ({k}-Fold)")
        ax.set_xlabel("Acurácia")
        ax.set_ylabel("Frequência")
        save_and_show(fig, "kfold_histogram.png")

        # 📊 Boxplot das acurácias
        fig, ax = plt.subplots()
        ax.boxplot(scores, vert=False)
        ax.set_title(f"Boxplot das acurácias ({k}-Fold)")
        ax.set_xlabel("Acurácia")
        save_and_show(fig, "kfold_boxplot.png")


def main():
    parser = argparse.ArgumentParser(description="Avaliação de classificadores")
    parser.add_argument("--dataset", type=str, default="abalone_dataset.csv")
    parser.add_argument(
        "--best-file", type=str, default="out/all_models/best_model.json"
    )
    parser.add_argument(
        "--method", type=str, choices=["holdout", "kfold"], default="holdout"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Número de folds (se usar kfold)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Exibir e salvar gráficos de avaliação"
    )
    args = parser.parse_args()

    # Carregar dataset
    df = pd.read_csv(args.dataset)
    target_col, feature_cols, num_cols, cat_cols = infer_columns(df)
    X = df[feature_cols]
    y = df[target_col]

    # Carregar melhor modelo salvo
    best_file = Path(args.best_file)
    if not best_file.exists():
        raise FileNotFoundError(
            f"Arquivo {best_file} não encontrado. Rode all_models.py antes."
        )

    best_payload = json.loads(best_file.read_text())
    model_name = best_payload["model"]
    best_params = best_payload["best_params"]

    print(f"Carregando melhor modelo: {model_name}")
    print("Parâmetros:", best_params)

    # Reconstruir pipeline
    model, _ = get_model_grid(model_name)
    pipe = build_pipeline(model, num_cols, cat_cols)
    pipe.set_params(**best_params)

    # Avaliação
    if args.method == "holdout":
        evaluate_holdout(pipe, X, y, plot=args.plot)
    else:
        evaluate_kfold(pipe, X, y, k=args.k, plot=args.plot)


if __name__ == "__main__":
    main()
