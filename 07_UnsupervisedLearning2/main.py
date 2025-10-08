"""
Análise de Clusterização - Espessura Epitelial
Atividade 4 - Descoberta de Perfis/Padrões de Olhos

Este script executa a análise completa de clusterização usando K-Means
para identificar perfis de espessura epitelial em mapas oculares.
"""

from pathlib import Path
from kmeans_clustering import KMeansEpithelialClusterer
from optimization import KOptimizer
from presentation import ClientPresentation


def main():
    """
    Executa a análise completa de clusterização
    """
    # Configurações
    data_path = Path(__file__).parent / "data" / "RTVue_20221110_MLClass.csv"
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)

    print("=" * 80)
    print(" " * 20 + "ANÁLISE DE ESPESSURA EPITELIAL")
    print(" " * 15 + "Descoberta de Perfis/Padrões de Olhos")
    print("=" * 80)

    # Opções de análise
    print("\nEscolha a análise que deseja executar:")
    print("1. Análise Completa (Otimização + Clusterização + Apresentação)")
    print("2. Apenas Otimização do Número de Clusters")
    print("3. Apenas Clusterização com K-Means")
    print("4. Apenas Apresentação para Cliente")
    print("5. Análise Rápida (K=3 sem otimização)")

    choice = input("\nDigite o número da opção (padrão=5): ").strip() or "5"

    if choice == "1":
        # Análise completa
        print("\n" + "=" * 80)
        print("EXECUTANDO ANÁLISE COMPLETA")
        print("=" * 80)

        # 1. Otimização
        print("\n" + "-" * 80)
        print("ETAPA 1: OTIMIZAÇÃO DO NÚMERO DE CLUSTERS")
        print("-" * 80)
        optimizer = KOptimizer(data_path)
        metrics_df = optimizer.analyze_optimal_k(max_k=10)

        # 2. Pergunta ao usuário qual K usar
        k_choice = input(
            "\nCom base na análise, quantos clusters deseja usar? (padrão=3): "
        ).strip()
        k = int(k_choice) if k_choice else 3

        # 3. Clusterização
        print("\n" + "-" * 80)
        print(f"ETAPA 2: CLUSTERIZAÇÃO COM K={k}")
        print("-" * 80)
        clusterer = KMeansEpithelialClusterer(n_clusters=k, random_state=42)
        clusterer.load_data(data_path)
        clusterer.preprocess_data()
        clusterer.fit()
        cluster_stats, cluster_counts = clusterer.get_cluster_profiles()

        print("\n📊 Estatísticas dos Clusters:")
        print(cluster_stats.xs("mean", level=1, axis=1))

        clusterer.visualize_clusters(
            save_path=results_path / "kmeans_distributions.png"
        )
        clusterer.visualize_cluster_means(
            save_path=results_path / "kmeans_profiles.png"
        )
        clusterer.save_results(results_path / "kmeans_results.csv")

        # 4. Apresentação
        print("\n" + "-" * 80)
        print("ETAPA 3: GERANDO APRESENTAÇÃO PARA CLIENTE")
        print("-" * 80)
        presentation = ClientPresentation(n_clusters=k)
        presentation.setup_and_train(data_path)
        presentation.generate_full_report()

    elif choice == "2":
        # Apenas otimização
        optimizer = KOptimizer(data_path)
        metrics_df = optimizer.analyze_optimal_k(max_k=10)
        print("\n💾 Métricas salvas em: results/k_optimization_metrics.csv")
        metrics_df.to_csv(results_path / "k_optimization_metrics.csv", index=False)

    elif choice == "3":
        # Apenas clusterização
        k_input = input("Quantos clusters deseja usar? (padrão=3): ").strip()
        k = int(k_input) if k_input else 3

        clusterer = KMeansEpithelialClusterer(n_clusters=k, random_state=42)
        clusterer.load_data(data_path)
        clusterer.preprocess_data()
        clusterer.fit()
        cluster_stats, cluster_counts = clusterer.get_cluster_profiles()

        print("\n📊 Estatísticas dos Clusters:")
        print(cluster_stats.xs("mean", level=1, axis=1))

        clusterer.visualize_clusters(
            save_path=results_path / "kmeans_distributions.png"
        )
        clusterer.visualize_cluster_means(
            save_path=results_path / "kmeans_profiles.png"
        )
        clusterer.save_results(results_path / "kmeans_results.csv")

    elif choice == "4":
        # Apenas apresentação
        k_input = input("Quantos clusters foram usados? (padrão=3): ").strip()
        k = int(k_input) if k_input else 3

        presentation = ClientPresentation(n_clusters=k)
        presentation.setup_and_train(data_path)
        presentation.generate_full_report()

    else:  # choice == "5" ou qualquer outro
        # Análise rápida com K=3
        print("\n" + "=" * 80)
        print("EXECUTANDO ANÁLISE RÁPIDA (K=3)")
        print("=" * 80)

        clusterer = KMeansEpithelialClusterer(n_clusters=3, random_state=42)
        clusterer.load_data(data_path)
        clusterer.preprocess_data()
        clusterer.fit()
        cluster_stats, cluster_counts = clusterer.get_cluster_profiles()

        print("\n📊 Estatísticas dos Clusters:")
        print(cluster_stats.xs("mean", level=1, axis=1))

        clusterer.visualize_clusters(
            save_path=results_path / "kmeans_distributions.png"
        )
        clusterer.visualize_cluster_means(
            save_path=results_path / "kmeans_profiles.png"
        )
        clusterer.save_results(results_path / "kmeans_results.csv")

        # Apresentação
        presentation = ClientPresentation(n_clusters=3)
        presentation.setup_and_train(data_path)
        presentation.generate_full_report()

    print("\n" + "=" * 80)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Resultados salvos em: {results_path}")
    print("\nArquivos gerados:")
    print("  📊 Visualizações de clusters")
    print("  📈 Apresentações para o cliente")
    print("  💾 Resultados em CSV")


if __name__ == "__main__":
    main()
