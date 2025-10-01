import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from kmeans_clustering import KMeansClusterer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


class KMeansOptimizer:
    def __init__(self):
        """
        Classe para otimizar o número de clusters no K-Means
        """
        self.data_path = (
            Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
        )
        self.clusterer = None
        self.optimal_results = {}

    def comprehensive_k_analysis(self, max_k=15):
        """
        Análise abrangente para encontrar o k ótimo
        """
        print("🔍 ANÁLISE ABRANGENTE DO NÚMERO ÓTIMO DE CLUSTERS (K)")
        print("=" * 60)

        # Inicializa clusterer para análise
        self.clusterer = KMeansClusterer()
        self.clusterer.load_data(self.data_path)
        self.clusterer.preprocess_data()

        # Testa diferentes valores de k
        print(f"Testando k de 2 a {max_k}...")
        metrics = self.clusterer.find_optimal_clusters(max_clusters=max_k)

        # Converte para DataFrame para análise
        metrics_df = pd.DataFrame(metrics)

        return metrics_df

    def find_optimal_k_methods(self, metrics_df):
        """
        Encontra k ótimo usando diferentes métodos
        """
        print(f"\n📊 MÉTODOS PARA ENCONTRAR K ÓTIMO:")
        print("=" * 50)

        # 1. Método do Cotovelo (Elbow Method)
        inertias = metrics_df["inertia"].values
        k_values = metrics_df["n_clusters"].values

        # Calcula diferenças (primeira derivada)
        diff1 = np.diff(inertias)
        # Calcula segunda derivada
        diff2 = np.diff(diff1)

        # Encontra o ponto de máxima curvatura
        elbow_k = k_values[np.argmax(np.abs(diff2)) + 2]

        print(f"🔄 MÉTODO DO COTOVELO:")
        print(f"   K ótimo sugerido: {elbow_k}")

        # 2. Melhor Silhouette Score
        best_silhouette_idx = metrics_df["silhouette"].idxmax()
        silhouette_k = metrics_df.loc[best_silhouette_idx, "n_clusters"]
        silhouette_score_val = metrics_df.loc[best_silhouette_idx, "silhouette"]

        print(f"📈 MELHOR SILHOUETTE SCORE:")
        print(f"   K ótimo: {silhouette_k}")
        print(f"   Silhouette Score: {silhouette_score_val:.4f}")

        # 3. Melhor Calinski-Harabasz
        best_ch_idx = metrics_df["calinski_harabasz"].idxmax()
        ch_k = metrics_df.loc[best_ch_idx, "n_clusters"]
        ch_score = metrics_df.loc[best_ch_idx, "calinski_harabasz"]

        print(f"🎯 MELHOR CALINSKI-HARABASZ:")
        print(f"   K ótimo: {ch_k}")
        print(f"   Score: {ch_score:.2f}")

        # 4. Melhor Davies-Bouldin (menor é melhor)
        best_db_idx = metrics_df["davies_bouldin"].idxmin()
        db_k = metrics_df.loc[best_db_idx, "n_clusters"]
        db_score = metrics_df.loc[best_db_idx, "davies_bouldin"]

        print(f"📉 MELHOR DAVIES-BOULDIN:")
        print(f"   K ótimo: {db_k}")
        print(f"   Score: {db_score:.4f}")

        # 5. Análise de estabilidade (diminuição da inércia)
        inertia_reduction = []
        for i in range(1, len(inertias)):
            reduction = (inertias[i - 1] - inertias[i]) / inertias[i - 1] * 100
            inertia_reduction.append(reduction)

        # Encontra onde a redução fica menor que 10%
        stability_threshold = 10  # 10% de redução
        stability_k = None
        for i, reduction in enumerate(inertia_reduction):
            if reduction < stability_threshold:
                stability_k = k_values[i + 1]
                break

        print(f"⚖️  ANÁLISE DE ESTABILIDADE:")
        print(f"   K onde redução < {stability_threshold}%: {stability_k}")

        return {
            "elbow": elbow_k,
            "silhouette": silhouette_k,
            "calinski_harabasz": ch_k,
            "davies_bouldin": db_k,
            "stability": stability_k,
            "metrics_df": metrics_df,
        }

    def visualize_k_optimization(self, metrics_df, save_path=None):
        """
        Visualiza a análise de otimização do k
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "OTIMIZAÇÃO DO NÚMERO DE CLUSTERS (K) - K-MEANS\nAnálise Completa para Escolha do K Ótimo",
            fontsize=16,
            fontweight="bold",
        )

        k_values = metrics_df["n_clusters"]

        # 1. Método do Cotovelo - Inércia
        ax1.plot(k_values, metrics_df["inertia"], "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Número de Clusters (k)")
        ax1.set_ylabel("Inércia (Within-cluster Sum of Squares)")
        ax1.set_title("Método do Cotovelo\n(Procurar ponto de inflexão)")
        ax1.grid(True, alpha=0.3)

        # Destacar k=3
        k3_inertia = metrics_df[metrics_df["n_clusters"] == 3]["inertia"].iloc[0]
        ax1.scatter(
            [3],
            [k3_inertia],
            color="red",
            s=150,
            marker="s",
            label="k=3 (atual)",
            zorder=5,
        )
        ax1.legend()

        # 2. Silhouette Score
        ax2.plot(k_values, metrics_df["silhouette"], "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("Número de Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score por K\n(Maior é melhor)")
        ax2.grid(True, alpha=0.3)

        # Destacar melhor k e k=3
        best_k = metrics_df.loc[metrics_df["silhouette"].idxmax(), "n_clusters"]
        best_score = metrics_df["silhouette"].max()
        ax2.scatter(
            [best_k],
            [best_score],
            color="gold",
            s=150,
            marker="*",
            label=f"Melhor k={best_k}",
            zorder=5,
        )

        k3_silhouette = metrics_df[metrics_df["n_clusters"] == 3]["silhouette"].iloc[0]
        ax2.scatter(
            [3],
            [k3_silhouette],
            color="red",
            s=150,
            marker="s",
            label="k=3 (atual)",
            zorder=5,
        )
        ax2.legend()

        # 3. Calinski-Harabasz Score
        ax3.plot(
            k_values, metrics_df["calinski_harabasz"], "mo-", linewidth=2, markersize=8
        )
        ax3.set_xlabel("Número de Clusters (k)")
        ax3.set_ylabel("Calinski-Harabasz Score")
        ax3.set_title("Calinski-Harabasz Score por K\n(Maior é melhor)")
        ax3.grid(True, alpha=0.3)

        # Destacar melhor k e k=3
        best_ch_k = metrics_df.loc[
            metrics_df["calinski_harabasz"].idxmax(), "n_clusters"
        ]
        best_ch_score = metrics_df["calinski_harabasz"].max()
        ax3.scatter(
            [best_ch_k],
            [best_ch_score],
            color="gold",
            s=150,
            marker="*",
            label=f"Melhor k={best_ch_k}",
            zorder=5,
        )

        k3_ch = metrics_df[metrics_df["n_clusters"] == 3]["calinski_harabasz"].iloc[0]
        ax3.scatter(
            [3], [k3_ch], color="red", s=150, marker="s", label="k=3 (atual)", zorder=5
        )
        ax3.legend()

        # 4. Davies-Bouldin Score
        ax4.plot(
            k_values, metrics_df["davies_bouldin"], "co-", linewidth=2, markersize=8
        )
        ax4.set_xlabel("Número de Clusters (k)")
        ax4.set_ylabel("Davies-Bouldin Score")
        ax4.set_title("Davies-Bouldin Score por K\n(Menor é melhor)")
        ax4.grid(True, alpha=0.3)

        # Destacar melhor k e k=3
        best_db_k = metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "n_clusters"]
        best_db_score = metrics_df["davies_bouldin"].min()
        ax4.scatter(
            [best_db_k],
            [best_db_score],
            color="gold",
            s=150,
            marker="*",
            label=f"Melhor k={best_db_k}",
            zorder=5,
        )

        k3_db = metrics_df[metrics_df["n_clusters"] == 3]["davies_bouldin"].iloc[0]
        ax4.scatter(
            [3], [k3_db], color="red", s=150, marker="s", label="k=3 (atual)", zorder=5
        )
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def compare_k_values(self, k_candidates):
        """
        Compara diferentes valores de k em detalhes
        """
        print(f"\n🔬 COMPARAÇÃO DETALHADA DE VALORES DE K:")
        print("=" * 60)

        results = {}

        for k in k_candidates:
            print(f"\n📊 TESTANDO K = {k}:")
            print("-" * 30)

            # Cria clusterer para este k
            clusterer = KMeansClusterer(n_clusters=k)
            clusterer.load_data(self.data_path)
            clusterer.preprocess_data()
            clusterer.fit()

            # Avalia
            evaluation = clusterer.evaluate()
            cluster_stats, cluster_counts = clusterer.get_cluster_profiles()

            # Armazena resultados
            results[k] = {
                "evaluation": evaluation,
                "cluster_counts": cluster_counts,
                "cluster_stats": cluster_stats,
                "clusterer": clusterer,
            }

            # Mostra métricas
            print(f"Silhouette Score: {evaluation['silhouette_score']:.4f}")
            print(f"Calinski-Harabasz: {evaluation['calinski_harabasz_score']:.2f}")
            print(f"Davies-Bouldin: {evaluation['davies_bouldin_score']:.4f}")
            print(f"Inércia: {evaluation['inertia']:.2f}")

            # Mostra distribuição
            print(f"Distribuição por cluster:")
            for cluster_id, count in cluster_counts.items():
                percentage = (count / cluster_counts.sum()) * 100
                print(
                    f"  Cluster {cluster_id}: {count:4d} pacientes ({percentage:5.1f}%)"
                )

            # Analisa equilíbrio dos clusters
            cluster_sizes = cluster_counts.values
            std_cluster_size = np.std(cluster_sizes)
            mean_cluster_size = np.mean(cluster_sizes)
            cv_cluster_size = (
                std_cluster_size / mean_cluster_size
            )  # Coeficiente de variação

            print(f"Equilíbrio dos clusters (CV): {cv_cluster_size:.3f}")
            if cv_cluster_size < 0.3:
                print("  ✅ Clusters bem equilibrados")
            elif cv_cluster_size < 0.5:
                print("  ⚠️  Clusters moderadamente equilibrados")
            else:
                print("  ❌ Clusters desequilibrados")

        return results

    def clinical_interpretation_k_analysis(self, results):
        """
        Interpretação clínica dos diferentes valores de k
        """
        print(f"\n🏥 INTERPRETAÇÃO CLÍNICA DOS VALORES DE K:")
        print("=" * 60)

        for k, result in results.items():
            print(f"\n🎯 K = {k} CLUSTERS:")
            print("-" * 25)

            cluster_counts = result["cluster_counts"]
            evaluation = result["evaluation"]

            # Interpretação clínica
            if k == 2:
                print("👁️  INTERPRETAÇÃO: Divisão binária (Normal vs Anormal)")
                print("   Prós: Simples, fácil decisão clínica")
                print(
                    "   Contras: Perde nuances importantes entre hipermetropia e miopia"
                )

            elif k == 3:
                print("👁️  INTERPRETAÇÃO: Clássica (Hipermetropia, Emetropia, Miopia)")
                print(
                    "   Prós: Alinha com conhecimento oftalmológico, protocolos estabelecidos"
                )
                print("   Contras: Pode mascarar subgrupos importantes")

            elif k == 4:
                print("👁️  INTERPRETAÇÃO: Refinada (possivelmente com astigmatismo)")
                print("   Prós: Captura mais variabilidade, pode separar astigmatismo")
                print("   Contras: Complexidade operacional aumenta")

            elif k == 5:
                print("👁️  INTERPRETAÇÃO: Muito específica (subtipos por severidade)")
                print("   Prós: Máxima personalização de tratamentos")
                print("   Contras: Pode ser excessivamente granular")

            else:
                print(f"👁️  INTERPRETAÇÃO: Altamente específica ({k} subgrupos)")
                print("   Prós: Segmentação muito detalhada")
                print(
                    "   Contras: Complexidade operacional alta, pode haver overfitting"
                )

            # Viabilidade operacional
            min_cluster_size = min(cluster_counts.values)
            max_cluster_size = max(cluster_counts.values)

            print(f"\n📋 VIABILIDADE OPERACIONAL:")
            print(f"   Menor grupo: {min_cluster_size} pacientes")
            print(f"   Maior grupo: {max_cluster_size} pacientes")

            if min_cluster_size < 50:
                print("   ❌ Grupos muito pequenos - inviável operacionalmente")
            elif min_cluster_size < 100:
                print("   ⚠️  Grupos pequenos - viabilidade limitada")
            else:
                print("   ✅ Todos os grupos são operacionalmente viáveis")

    def final_k_recommendation(self, optimal_methods, results):
        """
        Recomendação final para o valor de k
        """
        print(f"\n🎯 RECOMENDAÇÃO FINAL PARA O VALOR DE K:")
        print("=" * 60)

        print(f"\n📊 RESUMO DOS MÉTODOS:")
        print(f"• Método do Cotovelo: k = {optimal_methods['elbow']}")
        print(f"• Melhor Silhouette: k = {optimal_methods['silhouette']}")
        print(f"• Melhor Calinski-Harabasz: k = {optimal_methods['calinski_harabasz']}")
        print(f"• Melhor Davies-Bouldin: k = {optimal_methods['davies_bouldin']}")
        print(f"• Estabilidade: k = {optimal_methods['stability']}")

        # Análise de consenso
        method_votes = [
            optimal_methods["elbow"],
            optimal_methods["silhouette"],
            optimal_methods["calinski_harabasz"],
            optimal_methods["davies_bouldin"],
        ]

        # Remove None values
        method_votes = [vote for vote in method_votes if vote is not None]

        if method_votes:
            from collections import Counter

            vote_counts = Counter(method_votes)
            consensus_k = vote_counts.most_common(1)[0][0]
            consensus_votes = vote_counts.most_common(1)[0][1]

            print(f"\n🗳️  CONSENSO DOS MÉTODOS:")
            print(f"   k = {consensus_k} (recebeu {consensus_votes} votos)")

        print(f"\n💡 ANÁLISE CONTEXTUAL:")

        # Compara k=3 com outros valores
        k3_metrics = results[3]["evaluation"]

        print(f"\n📈 K = 3 (ATUAL):")
        print(f"   Silhouette: {k3_metrics['silhouette_score']:.4f}")
        print(f"   Interpretação clínica: Excelente")
        print(f"   Viabilidade operacional: Excelente")

        # Encontra o melhor k por silhouette
        best_k_silhouette = optimal_methods["silhouette"]
        if best_k_silhouette != 3 and best_k_silhouette in results:
            best_metrics = results[best_k_silhouette]["evaluation"]
            improvement = (
                (best_metrics["silhouette_score"] - k3_metrics["silhouette_score"])
                / k3_metrics["silhouette_score"]
            ) * 100

            print(f"\n🚀 K = {best_k_silhouette} (MELHOR TÉCNICO):")
            print(f"   Silhouette: {best_metrics['silhouette_score']:.4f}")
            print(f"   Melhoria técnica: {improvement:+.1f}%")

            if improvement > 20:
                print(f"   ✅ Melhoria significativa - considere k={best_k_silhouette}")
            elif improvement > 10:
                print(f"   ⚠️  Melhoria moderada - avaliar custo/benefício")
            else:
                print(f"   ❌ Melhoria marginal - manter k=3")

        print(f"\n🏆 RECOMENDAÇÃO FINAL:")

        if best_k_silhouette == 3:
            print(f"   ✅ MANTER K = 3")
            print(f"   Razão: Já é o valor ótimo técnico e clinicamente")
        else:
            k3_silhouette = k3_metrics["silhouette_score"]
            best_silhouette = results[best_k_silhouette]["evaluation"][
                "silhouette_score"
            ]
            improvement = ((best_silhouette - k3_silhouette) / k3_silhouette) * 100

            if improvement > 15 and best_k_silhouette <= 5:
                print(f"   🔄 CONSIDERAR K = {best_k_silhouette}")
                print(f"   Razão: Melhoria técnica significativa ({improvement:+.1f}%)")
                print(f"   Recomendação: Teste piloto com k={best_k_silhouette}")
            else:
                print(f"   ✅ MANTER K = 3")
                print(f"   Razão: Melhor equilíbrio técnico-clínico-operacional")


def main():
    """
    Função principal para otimização do k no K-Means
    """
    print("🔧 OTIMIZAÇÃO DO NÚMERO DE CLUSTERS (K) - K-MEANS")
    print("Análise para encontrar o k ótimo para dados oftalmológicos...")

    # Inicializa otimizador
    optimizer = KMeansOptimizer()

    # Análise abrangente
    metrics_df = optimizer.comprehensive_k_analysis(max_k=10)

    # Encontra k ótimo pelos diferentes métodos
    optimal_methods = optimizer.find_optimal_k_methods(metrics_df)

    # Visualiza análise
    results_path = Path(__file__).parent / "k_optimization_results"
    results_path.mkdir(exist_ok=True)

    optimizer.visualize_k_optimization(
        metrics_df, save_path=results_path / "k_means_k_optimization.png"
    )

    # Compara valores candidatos
    k_candidates = [2, 3, 4, 5, 6]  # Valores mais relevantes
    detailed_results = optimizer.compare_k_values(k_candidates)

    # Interpretação clínica
    optimizer.clinical_interpretation_k_analysis(detailed_results)

    # Recomendação final
    optimizer.final_k_recommendation(optimal_methods, detailed_results)

    # Salva resultados
    metrics_df.to_csv(results_path / "k_optimization_metrics.csv", index=False)

    print(f"\n✅ Análise de otimização do k concluída!")
    print(f"📁 Resultados salvos em: {results_path}")


if __name__ == "__main__":
    main()
