import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from kmeans_clustering import KMeansClusterer
import warnings

warnings.filterwarnings("ignore")


class ClinicalKComparison:
    def __init__(self):
        """
        Comparação clínica detalhada entre k=2 e k=3
        """
        self.data_path = (
            Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
        )

    def detailed_clinical_comparison(self):
        """
        Comparação clínica detalhada entre k=2 e k=3
        """
        print("🏥 COMPARAÇÃO CLÍNICA DETALHADA: K=2 vs K=3")
        print("=" * 60)

        results = {}

        for k in [2, 3]:
            print(f"\n📊 ANÁLISE DETALHADA - K = {k}:")
            print("-" * 40)

            # Cria clusterer
            clusterer = KMeansClusterer(n_clusters=k)
            clusterer.load_data(self.data_path)
            clusterer.preprocess_data()
            clusterer.fit()

            # Obtém perfis
            cluster_stats, cluster_counts = clusterer.get_cluster_profiles()
            evaluation = clusterer.evaluate()

            results[k] = {
                "clusterer": clusterer,
                "cluster_stats": cluster_stats,
                "cluster_counts": cluster_counts,
                "evaluation": evaluation,
            }

            # Análise por cluster
            for cluster_id in cluster_counts.index:
                count = cluster_counts[cluster_id]
                percentage = (count / cluster_counts.sum()) * 100

                al_mean = cluster_stats.loc[cluster_id, ("AL", "mean")]
                k1_mean = cluster_stats.loc[cluster_id, ("K1", "mean")]
                k2_mean = cluster_stats.loc[cluster_id, ("K2", "mean")]
                acd_mean = cluster_stats.loc[cluster_id, ("ACD", "mean")]

                print(f"\n🎯 Cluster {cluster_id}:")
                print(f"   Pacientes: {count} ({percentage:.1f}%)")
                print(f"   AL médio: {al_mean:.2f}mm")
                print(f"   K1 médio: {k1_mean:.2f}D")
                print(f"   K2 médio: {k2_mean:.2f}D")
                print(f"   ACD médio: {acd_mean:.2f}mm")

                # Interpretação clínica
                if al_mean < 23.0:
                    eye_type = "Olhos CURTOS (Hipermetrópicos)"
                elif al_mean > 24.5:
                    eye_type = "Olhos LONGOS (Miópicos)"
                else:
                    eye_type = "Olhos NORMAIS (Emetrópicos)"

                if k1_mean > 44.5:
                    cornea_curve = "Córnea CURVA (alto poder refrativo)"
                elif k1_mean < 42.5:
                    cornea_curve = "Córnea PLANA (baixo poder refrativo)"
                else:
                    cornea_curve = "Córnea NORMAL"

                print(f"   Tipo: {eye_type}")
                print(f"   Córnea: {cornea_curve}")

                # Astigmatismo
                astigmatism = abs(k1_mean - k2_mean)
                if astigmatism > 1.5:
                    astig_level = "ALTO astigmatismo"
                elif astigmatism > 0.75:
                    astig_level = "MODERADO astigmatismo"
                else:
                    astig_level = "BAIXO astigmatismo"

                print(f"   Astigmatismo: {astigmatism:.2f}D ({astig_level})")

        return results

    def visualize_k_comparison(self, results, save_path=None):
        """
        Visualiza comparação entre k=2 e k=3
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "COMPARAÇÃO CLÍNICA: K=2 vs K=3\nAnálise de Segmentação Oftalmológica",
            fontsize=16,
            fontweight="bold",
        )

        colors_k2 = ["#FF6B6B", "#4ECDC4"]
        colors_k3 = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Dados
        data = results[2]["clusterer"].features_data
        labels_k2 = results[2]["clusterer"].labels
        labels_k3 = results[3]["clusterer"].labels

        # 1. K=2 - AL vs K1
        ax1 = axes[0, 0]
        for i in range(2):
            mask = labels_k2 == i
            ax1.scatter(
                data.loc[mask, "AL"],
                data.loc[mask, "K1"],
                c=colors_k2[i],
                alpha=0.7,
                s=20,
                label=f"Grupo {i+1}",
            )
        ax1.set_xlabel("Comprimento Axial (AL) - mm")
        ax1.set_ylabel("Curvatura K1 - Dioptrias")
        ax1.set_title("K=2: Divisão Binária")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. K=3 - AL vs K1
        ax2 = axes[0, 1]
        for i in range(3):
            mask = labels_k3 == i
            ax2.scatter(
                data.loc[mask, "AL"],
                data.loc[mask, "K1"],
                c=colors_k3[i],
                alpha=0.7,
                s=20,
                label=f"Grupo {i+1}",
            )
        ax2.set_xlabel("Comprimento Axial (AL) - mm")
        ax2.set_ylabel("Curvatura K1 - Dioptrias")
        ax2.set_title("K=3: Divisão Clássica")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Distribuição de tamanhos
        ax3 = axes[0, 2]
        k2_counts = results[2]["cluster_counts"]
        k3_counts = results[3]["cluster_counts"]

        x_k2 = range(len(k2_counts))
        x_k3 = range(len(k3_counts))

        ax3.bar(
            [x - 0.2 for x in x_k2],
            k2_counts.values,
            width=0.4,
            label="K=2",
            alpha=0.7,
            color="blue",
        )
        ax3.bar(
            [x + 0.2 for x in x_k3],
            k3_counts.values,
            width=0.4,
            label="K=3",
            alpha=0.7,
            color="green",
        )

        ax3.set_xlabel("Clusters")
        ax3.set_ylabel("Número de Pacientes")
        ax3.set_title("Distribuição por Cluster")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Métricas de qualidade
        ax4 = axes[1, 0]

        metrics = ["Silhouette", "Calinski-H/100", "1-Davies-B"]
        k2_values = [
            results[2]["evaluation"]["silhouette_score"],
            results[2]["evaluation"]["calinski_harabasz_score"] / 100,
            1 - results[2]["evaluation"]["davies_bouldin_score"],
        ]
        k3_values = [
            results[3]["evaluation"]["silhouette_score"],
            results[3]["evaluation"]["calinski_harabasz_score"] / 100,
            1 - results[3]["evaluation"]["davies_bouldin_score"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax4.bar(x - width / 2, k2_values, width, label="K=2", alpha=0.7)
        ax4.bar(x + width / 2, k3_values, width, label="K=3", alpha=0.7)

        ax4.set_xlabel("Métricas")
        ax4.set_ylabel("Score (normalizado)")
        ax4.set_title("Qualidade do Clustering")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Análise de astigmatismo
        ax5 = axes[1, 1]

        # Calcula astigmatismo para cada grupo
        data_with_k2 = data.copy()
        data_with_k2["Cluster"] = labels_k2
        data_with_k2["Astigmatism"] = abs(data_with_k2["K1"] - data_with_k2["K2"])

        data_with_k3 = data.copy()
        data_with_k3["Cluster"] = labels_k3
        data_with_k3["Astigmatism"] = abs(data_with_k3["K1"] - data_with_k3["K2"])

        # Boxplot K=2
        k2_astig_data = [
            data_with_k2[data_with_k2["Cluster"] == i]["Astigmatism"].values
            for i in range(2)
        ]
        bp1 = ax5.boxplot(
            k2_astig_data, positions=[1, 2], widths=0.3, patch_artist=True
        )
        for patch, color in zip(bp1["boxes"], colors_k2):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Boxplot K=3
        k3_astig_data = [
            data_with_k3[data_with_k3["Cluster"] == i]["Astigmatism"].values
            for i in range(3)
        ]
        bp2 = ax5.boxplot(
            k3_astig_data, positions=[4, 5, 6], widths=0.3, patch_artist=True
        )
        for patch, color in zip(bp2["boxes"], colors_k3):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax5.set_xlabel("Clusters")
        ax5.set_ylabel("Astigmatismo (Dioptrias)")
        ax5.set_title("Distribuição de Astigmatismo")
        ax5.set_xticks([1.5, 5])
        ax5.set_xticklabels(["K=2", "K=3"])
        ax5.grid(True, alpha=0.3)

        # 6. Análise de protocolos clínicos
        ax6 = axes[1, 2]
        ax6.axis("off")

        protocol_text = """
        📋 PROTOCOLOS CLÍNICOS
        
        K=2 (BINÁRIO):
        ✅ Grupo 1: Olhos Curtos/Normais
           • Protocolos preventivos + hipermetropia
           
        ✅ Grupo 2: Olhos Longos
           • Protocolos para miopia
        
        K=3 (CLÁSSICO):
        ✅ Grupo 1: Hipermétropes
           • Monitoramento PIO + lentes divergentes
           
        ✅ Grupo 2: Míopes  
           • Controle progressão + lentes convergentes
           
        ✅ Grupo 3: Emetrópicos
           • Prevenção + acompanhamento
        
        COMPLEXIDADE:
        • K=2: 2 protocolos (mais simples)
        • K=3: 3 protocolos (mais específicos)
        """

        ax6.text(
            0.05,
            0.95,
            protocol_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def business_impact_analysis(self, results):
        """
        Análise do impacto comercial
        """
        print(f"\n💼 ANÁLISE DE IMPACTO COMERCIAL:")
        print("=" * 50)

        # Simulação de receita por tipo de tratamento
        revenue_per_group = {
            "hyperopic": 1500,  # Hipermétropes - monitoramento alto valor
            "myopic": 1200,  # Míopes - controle progressão
            "emmetropic": 800,  # Emetrópicos - prevenção
            "mixed": 1000,  # Grupo misto
        }

        print(f"\n📊 K = 2 (DIVISÃO BINÁRIA):")
        k2_counts = results[2]["cluster_counts"]
        k2_stats = results[2]["cluster_stats"]

        total_revenue_k2 = 0
        for cluster_id, count in k2_counts.items():
            al_mean = k2_stats.loc[cluster_id, ("AL", "mean")]

            if al_mean < 24.0:  # Olhos curtos/normais
                revenue_per_patient = revenue_per_group["mixed"]
                group_type = "Curtos/Normais (protocolo misto)"
            else:  # Olhos longos
                revenue_per_patient = revenue_per_group["myopic"]
                group_type = "Longos (protocolo miopia)"

            cluster_revenue = count * revenue_per_patient
            total_revenue_k2 += cluster_revenue

            print(f"   Cluster {cluster_id}: {count} pacientes")
            print(f"   Tipo: {group_type}")
            print(f"   Receita: R$ {cluster_revenue:,.0f}")

        print(f"   TOTAL K=2: R$ {total_revenue_k2:,.0f}")

        print(f"\n📊 K = 3 (DIVISÃO CLÁSSICA):")
        k3_counts = results[3]["cluster_counts"]
        k3_stats = results[3]["cluster_stats"]

        total_revenue_k3 = 0
        for cluster_id, count in k3_counts.items():
            al_mean = k3_stats.loc[cluster_id, ("AL", "mean")]

            if al_mean < 23.0:  # Hipermétropes
                revenue_per_patient = revenue_per_group["hyperopic"]
                group_type = "Hipermétropes (alto valor)"
            elif al_mean > 24.5:  # Míopes
                revenue_per_patient = revenue_per_group["myopic"]
                group_type = "Míopes (progressão)"
            else:  # Emetrópicos
                revenue_per_patient = revenue_per_group["emmetropic"]
                group_type = "Emetrópicos (prevenção)"

            cluster_revenue = count * revenue_per_patient
            total_revenue_k3 += cluster_revenue

            print(f"   Cluster {cluster_id}: {count} pacientes")
            print(f"   Tipo: {group_type}")
            print(f"   Receita: R$ {cluster_revenue:,.0f}")

        print(f"   TOTAL K=3: R$ {total_revenue_k3:,.0f}")

        # Comparação
        revenue_difference = total_revenue_k3 - total_revenue_k2
        percentage_diff = (revenue_difference / total_revenue_k2) * 100

        print(f"\n💰 COMPARAÇÃO FINANCEIRA:")
        print(f"   Diferença: R$ {revenue_difference:,.0f}")
        print(f"   Percentual: {percentage_diff:+.1f}%")

        if revenue_difference > 0:
            print(f"   ✅ K=3 gera maior receita")
        else:
            print(f"   ⚠️  K=2 gera maior receita")

    def final_recommendation(self, results):
        """
        Recomendação final considerando todos os aspectos
        """
        print(f"\n🎯 RECOMENDAÇÃO FINAL INTEGRADA:")
        print("=" * 50)

        k2_silhouette = results[2]["evaluation"]["silhouette_score"]
        k3_silhouette = results[3]["evaluation"]["silhouette_score"]
        improvement = ((k2_silhouette - k3_silhouette) / k3_silhouette) * 100

        print(f"\n📊 RESUMO TÉCNICO:")
        print(f"   K=2 Silhouette: {k2_silhouette:.4f}")
        print(f"   K=3 Silhouette: {k3_silhouette:.4f}")
        print(f"   Melhoria K=2: {improvement:+.1f}%")

        print(f"\n🏥 RESUMO CLÍNICO:")
        print(f"   K=2: Divisão simples (Normal vs Anormal)")
        print(f"   K=3: Divisão clássica (Hiper, Emétro, Míope)")

        print(f"\n💼 RESUMO OPERACIONAL:")
        print(f"   K=2: 2 protocolos (mais simples)")
        print(f"   K=3: 3 protocolos (mais específicos)")

        print(f"\n🏆 DECISÃO FINAL:")

        if improvement > 25:  # Melhoria muito significativa
            print(f"   🔄 RECOMENDO MIGRAR PARA K=2")
            print(f"   Razões:")
            print(f"   ✅ Melhoria técnica muito significativa ({improvement:+.1f}%)")
            print(f"   ✅ Menor complexidade operacional")
            print(f"   ✅ Implementação mais simples")
            print(f"   ⚠️  Perda de granularidade clínica aceitável")

        elif improvement > 15:  # Melhoria significativa
            print(f"   ⚖️  CONSIDERAR MIGRAÇÃO PARA K=2")
            print(f"   Razões:")
            print(f"   ✅ Melhoria técnica significativa ({improvement:+.1f}%)")
            print(f"   ⚠️  Avaliar trade-off: simplicidade vs especificidade")
            print(f"   💡 Sugestão: Teste A/B com ambas abordagens")

        else:  # Melhoria marginal
            print(f"   ✅ MANTER K=3")
            print(f"   Razões:")
            print(f"   ✅ Melhoria técnica marginal ({improvement:+.1f}%)")
            print(f"   ✅ Maior especificidade clínica")
            print(f"   ✅ Alinhamento com conhecimento oftalmológico")
            print(f"   ✅ Protocolos já estabelecidos")


def main():
    """
    Função principal para comparação clínica detalhada
    """
    print("🏥 ANÁLISE CLÍNICA COMPARATIVA: K=2 vs K=3")
    print("Avaliação técnica, clínica e comercial...")

    # Inicializa comparador
    comparator = ClinicalKComparison()

    # Comparação clínica detalhada
    results = comparator.detailed_clinical_comparison()

    # Visualização
    results_path = Path(__file__).parent / "k_optimization_results"
    results_path.mkdir(exist_ok=True)

    comparator.visualize_k_comparison(
        results, save_path=results_path / "clinical_k2_vs_k3_comparison.png"
    )

    # Análise de impacto comercial
    comparator.business_impact_analysis(results)

    # Recomendação final
    comparator.final_recommendation(results)

    print(f"\n✅ Análise clínica comparativa concluída!")
    print(f"📁 Resultados salvos em: {results_path}")


if __name__ == "__main__":
    main()
