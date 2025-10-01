import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dbscan_clustering import DBSCANClusterer
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


class DBSCANOptimizer:
    def __init__(self):
        """
        Classe para otimizar parâmetros do DBSCAN
        """
        self.data_path = (
            Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
        )
        self.clusterer = None

    def comprehensive_parameter_search(self):
        """
        Busca exaustiva dos melhores parâmetros
        """
        print("🔍 OTIMIZAÇÃO DE PARÂMETROS DBSCAN")
        print("=" * 50)

        # Inicializa clusterer para testar parâmetros
        self.clusterer = DBSCANClusterer()
        self.clusterer.load_data(self.data_path)
        self.clusterer.preprocess_data()

        # Ranges de parâmetros mais amplos
        eps_range = np.arange(0.3, 1.5, 0.1)  # 0.3 a 1.4 com step 0.1
        min_samples_range = range(3, 15)  # 3 a 14

        print(
            f"Testando {len(eps_range)} valores de eps e {len(min_samples_range)} valores de min_samples"
        )
        print(f"Total de combinações: {len(eps_range) * len(min_samples_range)}")

        # Executa busca
        results = self.clusterer.test_parameters(
            eps_range=eps_range.tolist(), min_samples_range=list(min_samples_range)
        )

        return results

    def analyze_optimal_configurations(self, results):
        """
        Analisa as melhores configurações encontradas
        """
        print(f"\n📊 ANÁLISE DOS RESULTADOS:")
        print("=" * 50)

        # Filtra resultados válidos
        valid_results = results.dropna(subset=["silhouette"])

        if len(valid_results) == 0:
            print("❌ Nenhuma configuração válida encontrada!")
            return None

        print(f"Configurações válidas encontradas: {len(valid_results)}")

        # 1. Melhores por silhouette score
        print(f"\n🏆 TOP 10 CONFIGURAÇÕES POR SILHOUETTE SCORE:")
        top_silhouette = valid_results.nlargest(10, "silhouette")
        for i, (idx, row) in enumerate(top_silhouette.iterrows(), 1):
            print(
                f"{i:2d}. eps={row['eps']:.1f}, min_samples={int(row['min_samples']):2d} → "
                f"Silhouette={row['silhouette']:.4f}, Clusters={int(row['n_clusters']):2d}, "
                f"Ruído={row['noise_ratio']*100:5.1f}%"
            )

        # 2. Configurações com exatamente 3 clusters
        three_clusters = valid_results[valid_results["n_clusters"] == 3]
        if len(three_clusters) > 0:
            print(f"\n🎯 CONFIGURAÇÕES COM 3 CLUSTERS (IDEAL CLÍNICO):")
            three_clusters_sorted = three_clusters.nlargest(5, "silhouette")
            for i, (idx, row) in enumerate(three_clusters_sorted.iterrows(), 1):
                print(
                    f"{i:2d}. eps={row['eps']:.1f}, min_samples={int(row['min_samples']):2d} → "
                    f"Silhouette={row['silhouette']:.4f}, Ruído={row['noise_ratio']*100:5.1f}%"
                )

            best_3_clusters = three_clusters_sorted.iloc[0]
            print(f"\n⭐ MELHOR CONFIGURAÇÃO PARA 3 CLUSTERS:")
            print(
                f"   eps={best_3_clusters['eps']}, min_samples={int(best_3_clusters['min_samples'])}"
            )
            print(f"   Silhouette Score: {best_3_clusters['silhouette']:.4f}")
            print(f"   Pontos de ruído: {best_3_clusters['noise_ratio']*100:.1f}%")

            return best_3_clusters
        else:
            print(f"\n❌ Nenhuma configuração resultou em exatamente 3 clusters")
            best_overall = valid_results.iloc[0]
            print(f"\n⭐ MELHOR CONFIGURAÇÃO GERAL:")
            print(
                f"   eps={best_overall['eps']}, min_samples={int(best_overall['min_samples'])}"
            )
            print(f"   Clusters: {int(best_overall['n_clusters'])}")
            print(f"   Silhouette Score: {best_overall['silhouette']:.4f}")
            print(f"   Pontos de ruído: {best_overall['noise_ratio']*100:.1f}%")

            return best_overall

    def compare_configurations(self, results):
        """
        Compara configuração original vs otimizada
        """
        print(f"\n🔄 COMPARAÇÃO: ORIGINAL vs OTIMIZADA")
        print("=" * 50)

        # Configuração original
        original_config = results[
            (results["eps"] == 0.5) & (results["min_samples"] == 5)
        ]

        if len(original_config) > 0:
            orig = original_config.iloc[0]
            print(f"📌 CONFIGURAÇÃO ORIGINAL (eps=0.5, min_samples=5):")
            print(f"   Clusters: {int(orig['n_clusters'])}")
            print(
                f"   Silhouette: {orig['silhouette']:.4f}"
                if not pd.isna(orig["silhouette"])
                else "   Silhouette: N/A"
            )
            print(f"   Pontos de ruído: {orig['noise_ratio']*100:.1f}%")

        # Melhor configuração
        valid_results = results.dropna(subset=["silhouette"])
        if len(valid_results) > 0:
            best = valid_results.loc[valid_results["silhouette"].idxmax()]
            print(f"\n🚀 MELHOR CONFIGURAÇÃO ENCONTRADA:")
            print(f"   eps={best['eps']}, min_samples={int(best['min_samples'])}")
            print(f"   Clusters: {int(best['n_clusters'])}")
            print(f"   Silhouette: {best['silhouette']:.4f}")
            print(f"   Pontos de ruído: {best['noise_ratio']*100:.1f}%")

            if len(original_config) > 0 and not pd.isna(orig["silhouette"]):
                improvement = (
                    (best["silhouette"] - orig["silhouette"]) / orig["silhouette"]
                ) * 100
                noise_reduction = orig["noise_ratio"] - best["noise_ratio"]
                print(f"\n📈 MELHORIAS:")
                print(f"   Silhouette Score: {improvement:+.1f}%")
                print(
                    f"   Redução de ruído: {noise_reduction*100:+.1f} pontos percentuais"
                )

            return best

        return None

    def visualize_parameter_space(self, results, save_path=None):
        """
        Visualiza o espaço de parâmetros
        """
        print(f"\n📊 Gerando visualização do espaço de parâmetros...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "OTIMIZAÇÃO DE PARÂMETROS DBSCAN\nAnálise do Espaço de Parâmetros",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Heatmap Silhouette Score
        valid_results = results.dropna(subset=["silhouette"])
        if len(valid_results) > 0:
            pivot_silhouette = valid_results.pivot(
                index="min_samples", columns="eps", values="silhouette"
            )
            sns.heatmap(pivot_silhouette, annot=True, fmt=".3f", cmap="viridis", ax=ax1)
            ax1.set_title("Silhouette Score por Parâmetros")
            ax1.set_xlabel("eps")
            ax1.set_ylabel("min_samples")

        # 2. Heatmap Número de Clusters
        pivot_clusters = results.pivot(
            index="min_samples", columns="eps", values="n_clusters"
        )
        sns.heatmap(pivot_clusters, annot=True, fmt="d", cmap="plasma", ax=ax2)
        ax2.set_title("Número de Clusters por Parâmetros")
        ax2.set_xlabel("eps")
        ax2.set_ylabel("min_samples")

        # 3. Heatmap Percentual de Ruído
        pivot_noise = results.pivot(
            index="min_samples", columns="eps", values="noise_ratio"
        )
        sns.heatmap(pivot_noise, annot=True, fmt=".2f", cmap="Reds", ax=ax3)
        ax3.set_title("Percentual de Ruído por Parâmetros")
        ax3.set_xlabel("eps")
        ax3.set_ylabel("min_samples")

        # 4. Scatter plot: Silhouette vs Noise Ratio
        if len(valid_results) > 0:
            scatter = ax4.scatter(
                valid_results["noise_ratio"],
                valid_results["silhouette"],
                c=valid_results["n_clusters"],
                cmap="tab10",
                alpha=0.7,
                s=50,
            )
            ax4.set_xlabel("Percentual de Ruído")
            ax4.set_ylabel("Silhouette Score")
            ax4.set_title("Trade-off: Qualidade vs Cobertura")
            plt.colorbar(scatter, ax=ax4, label="Número de Clusters")

            # Destacar configurações com 3 clusters
            three_clusters = valid_results[valid_results["n_clusters"] == 3]
            if len(three_clusters) > 0:
                ax4.scatter(
                    three_clusters["noise_ratio"],
                    three_clusters["silhouette"],
                    c="red",
                    marker="s",
                    s=100,
                    alpha=0.8,
                    label="3 Clusters",
                )
                ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def test_optimal_configuration(self, best_config):
        """
        Testa a configuração otimizada
        """
        print(f"\n🧪 TESTANDO CONFIGURAÇÃO OTIMIZADA:")
        print("=" * 40)

        # Cria novo clusterer com parâmetros otimizados
        optimized_clusterer = DBSCANClusterer(
            eps=float(best_config["eps"]), min_samples=int(best_config["min_samples"])
        )

        # Executa clustering
        optimized_clusterer.load_data(self.data_path)
        optimized_clusterer.preprocess_data()
        optimized_clusterer.fit()

        # Avalia resultados
        evaluation = optimized_clusterer.evaluate()
        cluster_stats, cluster_counts, n_noise = (
            optimized_clusterer.get_cluster_profiles()
        )

        print(
            f"✅ Configuração: eps={best_config['eps']}, min_samples={best_config['min_samples']}"
        )
        print(f"📊 Clusters encontrados: {evaluation['n_clusters']}")
        print(
            f"🎯 Pontos classificados: {len(optimized_clusterer.labels) - evaluation['n_noise']} ({((len(optimized_clusterer.labels) - evaluation['n_noise'])/len(optimized_clusterer.labels))*100:.1f}%)"
        )
        print(
            f"🔴 Pontos de ruído: {evaluation['n_noise']} ({evaluation['noise_ratio']*100:.1f}%)"
        )

        if not np.isnan(evaluation["silhouette_score"]):
            print(f"📈 Silhouette Score: {evaluation['silhouette_score']:.4f}")
            print(f"📊 Calinski-Harabasz: {evaluation['calinski_harabasz_score']:.2f}")
            print(f"📉 Davies-Bouldin: {evaluation['davies_bouldin_score']:.4f}")

        if cluster_counts is not None:
            print(f"\n📋 DISTRIBUIÇÃO POR CLUSTER:")
            for cluster_id, count in cluster_counts.items():
                percentage = (count / len(optimized_clusterer.labels)) * 100
                print(
                    f"   Cluster {cluster_id}: {count:4d} pacientes ({percentage:5.1f}%)"
                )

        # Visualiza resultados
        optimized_clusterer.plot_clusters_2d("AL", "K1")

        return optimized_clusterer

    def final_recommendation(self, original_eps, original_min_samples, best_config):
        """
        Recomendação final
        """
        print(f"\n🎯 RECOMENDAÇÃO FINAL:")
        print("=" * 50)

        print(f"💡 PARÂMETROS ORIGINAIS:")
        print(f"   eps = {original_eps}")
        print(f"   min_samples = {original_min_samples}")
        print(f"   Resultado: Múltiplos clusters pequenos + muito ruído")

        print(f"\n🚀 PARÂMETROS OTIMIZADOS:")
        print(f"   eps = {best_config['eps']}")
        print(f"   min_samples = {best_config['min_samples']}")
        print(
            f"   Resultado: {best_config['n_clusters']} clusters + {best_config['noise_ratio']*100:.1f}% ruído"
        )

        print(f"\n📊 IMPACTO DA OTIMIZAÇÃO:")
        if best_config["n_clusters"] == 3:
            print(f"   ✅ Conseguiu 3 clusters (clinicamente ideal)")
        else:
            print(f"   ⚠️  Resultou em {best_config['n_clusters']} clusters")

        print(f"   📈 Silhouette Score: {best_config['silhouette']:.4f}")
        print(
            f"   🎯 Cobertura: {(1-best_config['noise_ratio'])*100:.1f}% dos pacientes"
        )

        print(f"\n💭 CONCLUSÃO:")
        if best_config["noise_ratio"] < 0.2 and best_config["n_clusters"] == 3:
            print(f"   ✅ Otimização MUITO BEM-SUCEDIDA!")
            print(f"   ✅ DBSCAN otimizado pode ser uma alternativa viável ao K-Means")
        elif best_config["noise_ratio"] < 0.3:
            print(f"   ✅ Otimização bem-sucedida, mas ainda há limitações")
            print(f"   ⚠️  K-Means permanece superior para uso clínico")
        else:
            print(f"   ❌ Mesmo otimizado, DBSCAN não é adequado para este caso")
            print(f"   ❌ K-Means continua sendo a melhor escolha")


def main():
    """
    Função principal para otimização de parâmetros DBSCAN
    """
    print("🔧 OTIMIZAÇÃO DE PARÂMETROS DBSCAN")
    print("Buscando a melhor configuração para dados oftalmológicos...")

    # Inicializa otimizador
    optimizer = DBSCANOptimizer()

    # Busca exaustiva de parâmetros
    results = optimizer.comprehensive_parameter_search()

    # Analisa configurações ótimas
    best_config = optimizer.analyze_optimal_configurations(results)

    if best_config is not None:
        # Compara original vs otimizada
        optimizer.compare_configurations(results)

        # Visualiza espaço de parâmetros
        results_path = Path(__file__).parent / "optimization_results"
        results_path.mkdir(exist_ok=True)

        optimizer.visualize_parameter_space(
            results, save_path=results_path / "dbscan_parameter_optimization.png"
        )

        # Testa configuração otimizada
        optimized_clusterer = optimizer.test_optimal_configuration(best_config)

        # Salva resultados otimizados
        output_path = results_path / "dbscan_optimized_results.csv"
        optimized_clusterer.save_results(output_path)

        # Recomendação final
        optimizer.final_recommendation(0.5, 5, best_config)

        print(f"\n✅ Otimização concluída!")
        print(f"📁 Resultados salvos em: {results_path}")
    else:
        print(f"\n❌ Não foi possível encontrar configurações viáveis")
        print(f"🎯 Recomendação: Use K-Means com k=3")


if __name__ == "__main__":
    main()
