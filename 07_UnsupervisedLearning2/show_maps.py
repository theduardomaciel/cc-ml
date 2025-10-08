"""
Script para visualizar todos os mapas epiteliais gerados
"""

import os
from pathlib import Path


def show_generated_maps():
    """Mostra resumo de todos os mapas gerados"""

    print("\n" + "=" * 80)
    print("MAPAS EPITELIAIS GERADOS")
    print("=" * 80)

    # Mapas básicos
    print("\n📊 MAPAS BÁSICOS (results_epithelial_maps/)")
    basic_path = Path("results_epithelial_maps")
    if basic_path.exists():
        files = list(basic_path.glob("*.png"))
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"  ✓ {f.name} ({size_kb:.1f} KB)")

    # Mapas com remoção de outliers
    print(
        "\n📊 COM REMOÇÃO DE OUTLIERS (results_with_outlier_removal/epithelial_maps/)"
    )
    with_path = Path("results_with_outlier_removal/epithelial_maps")

    if with_path.exists():
        # Mapas de exemplo
        example_files = list(with_path.glob("*.png"))
        if example_files:
            print("\n  Amostras individuais:")
            for f in example_files:
                size_kb = f.stat().st_size / 1024
                print(f"    ✓ {f.name} ({size_kb:.1f} KB)")

        # Mapas por algoritmo
        for algo in ["kmeans", "dbscan", "kmedoids"]:
            algo_path = with_path / algo
            if algo_path.exists():
                files = list(algo_path.glob("*.png"))
                if files:
                    print(f"\n  {algo.upper()}:")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"    ✓ {f.name} ({size_kb:.1f} KB)")

    # Mapas sem remoção de outliers
    print(
        "\n📊 SEM REMOÇÃO DE OUTLIERS (results_without_outlier_removal/epithelial_maps/)"
    )
    without_path = Path("results_without_outlier_removal/epithelial_maps")

    if without_path.exists():
        # Mapas de exemplo
        example_files = list(without_path.glob("*.png"))
        if example_files:
            print("\n  Amostras individuais:")
            for f in example_files:
                size_kb = f.stat().st_size / 1024
                print(f"    ✓ {f.name} ({size_kb:.1f} KB)")

        # Mapas por algoritmo
        for algo in ["kmeans", "dbscan", "kmedoids"]:
            algo_path = without_path / algo
            if algo_path.exists():
                files = list(algo_path.glob("*.png"))
                if files:
                    print(f"\n  {algo.upper()}:")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"    ✓ {f.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)

    total_maps = 0
    for path in [basic_path, with_path, without_path]:
        if path.exists():
            total_maps += len(list(path.rglob("*.png")))

    print(f"\n✓ Total de mapas gerados: {total_maps}")
    print("✓ Para entender os mapas, leia: EPITHELIAL_MAPS.md")
    print("\nℹ️  Os mapas circulares mostram:")
    print("  - Centro (C) + 8 regiões periféricas (S, ST, T, IT, I, IN, N, SN)")
    print("  - Cores: 🔴 Vermelho = fino | 🟡 Amarelo = médio | 🟢 Verde = espesso")
    print("  - Mapas médios por cluster revelam padrões de cada grupo")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    show_generated_maps()
