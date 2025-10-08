import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from preprocessing import load_and_preprocess
from epithelial_mapping import create_cluster_average_maps, create_multiple_maps

sns.set_style("whitegrid")


def optimize_kmeans(scaled_data, max_k=10):
    """Encontra melhor k para K-Means"""
    metrics = {"k": [], "silhouette": [], "inertia": []}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        metrics["k"].append(k)
        metrics["silhouette"].append(silhouette_score(scaled_data, labels))
        metrics["inertia"].append(kmeans.inertia_)

    best_k = metrics["k"][np.argmax(metrics["silhouette"])]
    print(f"  Melhor k: {best_k} (Silhouette: {max(metrics['silhouette']):.4f})")
    return best_k, metrics


def optimize_dbscan(scaled_data):
    """Encontra melhor eps para DBSCAN"""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, _ = neighbors_fit.kneighbors(scaled_data)
    distances = np.sort(distances[:, -1])

    suggested_eps = np.percentile(distances, 95)
    print(f"  Eps sugerido: {suggested_eps:.3f}")

    # Testa alguns valores
    best_eps, best_min_samples = suggested_eps, 5
    best_score = -1

    for eps in [suggested_eps * 0.5, suggested_eps, suggested_eps * 1.5]:
        for min_samples in [3, 5, 10]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_data)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                score = silhouette_score(scaled_data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    print(f"  Melhores params: eps={best_eps:.3f}, min_samples={best_min_samples}")
    return best_eps, best_min_samples


def run_clustering(file_path, output_dir="results", remove_outliers=True):
    """
    Executa clustering com K-Means, DBSCAN e K-Medoids

    Args:
        file_path: Caminho do arquivo CSV
        output_dir: Diretório para salvar resultados
        remove_outliers: Se True, remove outliers antes do clustering
    """

    print("\n" + "=" * 80)
    print("CLUSTERING DE DADOS DE ESPESSURA EPITELIAL")
    print("=" * 80)

    # Preprocessamento
    print("\n[1/4] Pré-processamento")
    df, scaled_data, features, scaler = load_and_preprocess(
        file_path,
        remove_outliers=remove_outliers,
        outlier_method="iqr",
        iqr_multiplier=1.5,
    )

    results = {}
    best_k = 8  # Valor fixo para consistência
    # best_k, kmeans_metrics = optimize_kmeans(scaled_data)

    # K-Means
    print("\n[2/4] K-Means")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    results["kmeans"] = {
        "labels": kmeans_labels,
        "n_clusters": best_k,
        "silhouette": silhouette_score(scaled_data, kmeans_labels),
        "calinski": calinski_harabasz_score(scaled_data, kmeans_labels),
        "davies": davies_bouldin_score(scaled_data, kmeans_labels),
    }

    # DBSCAN
    print("\n[3/4] DBSCAN")
    best_eps, best_min_samples = optimize_dbscan(scaled_data)
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(scaled_data)

    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    if n_clusters >= 2:
        results["dbscan"] = {
            "labels": dbscan_labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": silhouette_score(scaled_data, dbscan_labels),
            "calinski": calinski_harabasz_score(scaled_data, dbscan_labels),
            "davies": davies_bouldin_score(scaled_data, dbscan_labels),
        }
    else:
        print(f"  ⚠ Apenas {n_clusters} cluster(s) encontrado(s)")
        results["dbscan"] = {"n_clusters": n_clusters, "n_noise": n_noise}

    # K-Medoids
    print("\n[4/4] K-Medoids")
    kmedoids = KMedoids(n_clusters=best_k, random_state=42)
    kmedoids_labels = kmedoids.fit_predict(scaled_data)

    results["kmedoids"] = {
        "labels": kmedoids_labels,
        "n_clusters": best_k,
        "silhouette": silhouette_score(scaled_data, kmedoids_labels),
        "calinski": calinski_harabasz_score(scaled_data, kmedoids_labels),
        "davies": davies_bouldin_score(scaled_data, kmedoids_labels),
    }

    # Salva resultados
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("RESUMO DOS RESULTADOS")
    print("=" * 80)

    summary_data = []
    for algo, res in results.items():
        print(f"\n{algo.upper()}:")
        print(f"  Clusters: {res.get('n_clusters', 'N/A')}")
        if "silhouette" in res:
            print(f"  Silhouette: {res['silhouette']:.4f}")
            print(f"  Calinski-Harabasz: {res['calinski']:.2f}")
            print(f"  Davies-Bouldin: {res['davies']:.4f}")

            # Distribuição
            unique, counts = np.unique(res["labels"], return_counts=True)
            for label, count in zip(unique, counts):
                pct = 100 * count / len(res["labels"])
                cluster_name = "Ruído" if label == -1 else f"Cluster {label}"
                print(f"    {cluster_name}: {count} ({pct:.1f}%)")

        summary_data.append(
            {
                "Algoritmo": algo,
                "Clusters": res.get("n_clusters", "N/A"),
                "Silhouette": res.get("silhouette", "N/A"),
                "Calinski-Harabasz": res.get("calinski", "N/A"),
                "Davies-Bouldin": res.get("davies", "N/A"),
            }
        )

    # Salva CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "clustering_results.csv", index=False)

    # Visualização comparativa
    plot_comparison(results, scaled_data, features, output_path)

    # Gera mapas epiteliais
    print("\n[BONUS] Gerando mapas epiteliais...")

    # Mapas de amostras aleatórias
    create_multiple_maps(
        df, output_dir=str(output_path / "epithelial_maps"), n_samples=6
    )

    # Mapas médios por cluster para cada algoritmo
    for algo in ["kmeans", "dbscan", "kmedoids"]:
        if algo in results and "labels" in results[algo]:
            print(f"  - Mapas do {algo.upper()}")
            create_cluster_average_maps(
                df,
                results[algo]["labels"],
                output_dir=str(output_path / "epithelial_maps" / algo),
            )

    print(f"\n✓ Resultados salvos em: {output_path.absolute()}")
    return results, df, scaled_data, features


def plot_comparison(results, scaled_data, features, output_path):
    """Cria visualização comparativa"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Comparação de Algoritmos de Clustering", fontsize=16, fontweight="bold"
    )

    algorithms = ["kmeans", "dbscan", "kmedoids"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for idx, (algo, color) in enumerate(zip(algorithms, colors)):
        if algo not in results or "labels" not in results[algo]:
            continue

        labels = results[algo]["labels"]

        # Distribuição de clusters
        ax = axes[0, idx]
        unique, counts = np.unique(labels, return_counts=True)
        bars = ax.bar(
            range(len(unique)), counts, color=color, alpha=0.7, edgecolor="black"
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Amostras")
        ax.set_title(
            f"{algo.upper()}\nSilhouette: {results[algo].get('silhouette', 'N/A'):.4f}"
        )
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels([f"{u}" if u != -1 else "Ruído" for u in unique])

        # Adiciona percentuais
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = 100 * count / len(labels)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Boxplot das features principais
        ax = axes[1, idx]
        df_plot = pd.DataFrame(scaled_data, columns=features)
        df_plot["Cluster"] = labels

        # Seleciona apenas primeiras 4 features para visualização
        selected_features = features[:4]
        data_to_plot = [
            df_plot[df_plot["Cluster"] == c][selected_features].values.flatten()
            for c in unique
            if c != -1
        ]

        if data_to_plot:
            bp = ax.boxplot(
                data_to_plot,
                tick_labels=[f"C{c}" for c in unique if c != -1],
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel("Valor Normalizado")
            ax.set_xlabel("Cluster")
            ax.set_title("Distribuição das Features")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "clustering_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Visualização salva")


if __name__ == "__main__":
    data_path = "data/RTVue_20221110_MLClass.csv"

    # Executa clustering COM remoção de outliers
    print("\n### COM REMOÇÃO DE OUTLIERS ###")
    results_with, df_with, scaled_with, features = run_clustering(
        data_path, output_dir="results_with_outlier_removal", remove_outliers=True
    )

    # Executa clustering SEM remoção de outliers
    print("\n\n### SEM REMOÇÃO DE OUTLIERS ###")
    results_without, df_without, scaled_without, _ = run_clustering(
        data_path, output_dir="results_without_outlier_removal", remove_outliers=False
    )
