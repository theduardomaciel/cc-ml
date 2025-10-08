"""
Script Principal - Análise de Perfis de Espessura Epitelial
Implementa a metodologia KDD completa
"""

from kmeans_clustering import KMeansEpithelialClusterer
from optimization import KOptimizer
from presentation import ClientPresentation
import sys


def print_menu():
    """Exibe o menu de opções."""
    print("\n" + "=" * 70)
    print("  ANÁLISE DE PERFIS DE ESPESSURA EPITELIAL - K-MEANS CLUSTERING")
    print("  Metodologia KDD (Knowledge Discovery in Databases)")
    print("=" * 70)
    print("\nEscolha uma opção:")
    print("  1. Análise Completa (Otimização + Clusterização + Apresentação)")
    print("  2. Apenas Otimização do Número de Clusters (K)")
    print("  3. Apenas Clusterização (com K específico)")
    print("  4. Apenas Gerar Visualizações de Apresentação")
    print("  5. Análise Rápida (K=3, sem otimização)")
    print("  0. Sair")
    print("=" * 70)


def run_optimization():
    """Executa apenas a otimização do número de clusters."""
    print("\n>>> MODO: OTIMIZAÇÃO DO NÚMERO DE CLUSTERS")
    
    # Carregar e preparar dados
    clusterer = KMeansEpithelialClusterer()
    df = clusterer.load_and_select_data('data/RTVue_20221110_MLClass.csv')
    df_clean = clusterer.preprocess_data(df)
    
    # Otimizar K
    optimizer = KOptimizer(k_range=(2, 11))
    df_metrics, k_optimal = optimizer.optimize(df_clean)
    optimizer.plot_optimization(df_metrics)
    optimizer.save_metrics(df_metrics)
    
    print(f"\n✓ Otimização concluída! K ótimo recomendado: {k_optimal}")
    return k_optimal


def run_clustering(n_clusters=3):
    """Executa apenas a clusterização."""
    print(f"\n>>> MODO: CLUSTERIZAÇÃO (K={n_clusters})")
    
    # Executar pipeline KDD completo
    clusterer = KMeansEpithelialClusterer(n_clusters=n_clusters)
    df_result, df_cluster_info, metrics = clusterer.fit('data/RTVue_20221110_MLClass.csv')
    clusterer.save_results(df_result)
    
    print("\n✓ Clusterização concluída!")
    return df_result, df_cluster_info, metrics


def run_presentation(df_result=None, df_cluster_info=None):
    """Executa apenas a geração de visualizações."""
    print("\n>>> MODO: GERAÇÃO DE VISUALIZAÇÕES")
    
    # Se não tiver dados, carregar do arquivo salvo
    if df_result is None:
        import pandas as pd
        try:
            df_result = pd.read_csv('results/kmeans_results.csv')
            print("✓ Resultados carregados de: results/kmeans_results.csv")
        except FileNotFoundError:
            print("❌ Erro: Arquivo results/kmeans_results.csv não encontrado.")
            print("   Execute a clusterização primeiro (opção 1, 3 ou 5).")
            return
    
    # Gerar visualizações
    presentation = ClientPresentation()
    presentation.generate_all(df_result, df_cluster_info)
    
    print("\n✓ Visualizações geradas com sucesso!")


def run_complete_analysis():
    """Executa a análise completa: otimização + clusterização + apresentação."""
    print("\n>>> MODO: ANÁLISE COMPLETA")
    print("\nEsta análise executará:")
    print("  1. Otimização do número de clusters")
    print("  2. Clusterização com K ótimo")
    print("  3. Geração de visualizações profissionais")
    
    # 1. Otimização
    print("\n" + "🔍" * 35)
    print("ETAPA 1/3: OTIMIZAÇÃO")
    print("🔍" * 35)
    k_optimal = run_optimization()
    
    # 2. Clusterização
    print("\n" + "⚙️" * 35)
    print("ETAPA 2/3: CLUSTERIZAÇÃO")
    print("⚙️" * 35)
    df_result, df_cluster_info, metrics = run_clustering(k_optimal)
    
    # 3. Apresentação
    print("\n" + "📊" * 35)
    print("ETAPA 3/3: VISUALIZAÇÕES")
    print("📊" * 35)
    run_presentation(df_result, df_cluster_info)
    
    print("\n" + "=" * 70)
    print("  ✅ ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
    print("=" * 70)
    print("\nArquivos gerados em 'results/':")
    print("  • k_optimization.png - Gráficos de otimização")
    print("  • k_optimization_metrics.csv - Métricas detalhadas")
    print("  • kmeans_results.csv - Resultados da clusterização")
    print("  • 01_executive_summary.png - Resumo executivo")
    print("  • 02_detailed_profiles.png - Análise detalhada")
    print("  • 03_clinical_interpretation.png - Interpretação clínica")
    print("  • kmeans_profiles.png - Perfis radiais")
    print("  • kmeans_distributions.png - Distribuições")


def run_quick_analysis():
    """Executa análise rápida com K=3."""
    print("\n>>> MODO: ANÁLISE RÁPIDA (K=3)")
    
    # Clusterização
    df_result, df_cluster_info, metrics = run_clustering(3)
    
    # Apresentação
    run_presentation(df_result, df_cluster_info)
    
    print("\n✓ Análise rápida concluída!")


def main():
    """Função principal."""
    while True:
        print_menu()
        
        try:
            choice = input("\nDigite sua opção: ").strip()
            
            if choice == '0':
                print("\n👋 Encerrando o programa. Até logo!")
                sys.exit(0)
            
            elif choice == '1':
                run_complete_analysis()
                input("\nPressione ENTER para continuar...")
            
            elif choice == '2':
                run_optimization()
                input("\nPressione ENTER para continuar...")
            
            elif choice == '3':
                k = int(input("\nDigite o número de clusters (K): "))
                if k < 2:
                    print("❌ Erro: K deve ser >= 2")
                    continue
                run_clustering(k)
                input("\nPressione ENTER para continuar...")
            
            elif choice == '4':
                run_presentation()
                input("\nPressione ENTER para continuar...")
            
            elif choice == '5':
                run_quick_analysis()
                input("\nPressione ENTER para continuar...")
            
            else:
                print("❌ Opção inválida! Tente novamente.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrompido pelo usuário. Até logo!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Erro: {str(e)}")
            input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    main()
