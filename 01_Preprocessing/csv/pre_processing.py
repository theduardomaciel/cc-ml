#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de pré-processamento conservador para dataset de diabetes.
Objetivo: Limpar os dados com o MÍNIMO de viés possível.
O script original de ML permanece INALTERADO.

Estratégia conservadora:
- Apenas trata valores claramente inválidos (zeros biologicamente impossíveis)
- Usa métodos estatísticos robustos para imputação
- NÃO remove outliers (podem ser casos reais)
- NÃO normaliza (deixa para o algoritmo decidir)
- Mantém a distribuição original dos dados o máximo possível

@author: Pré-processamento conservador
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os


def analyze_data_quality(df):
    """Analisa a qualidade dos dados de forma conservadora"""
    print("=" * 60)
    print("ANÁLISE DE QUALIDADE DOS DADOS")
    print("=" * 60)

    print(f"Shape do dataset: {df.shape}")
    print(f"Colunas: {list(df.columns)}")

    print(f"\nDados faltantes por coluna:")
    missing_data = df.isnull().sum()
    for col in df.columns:
        missing_count = missing_data[col]
        missing_pct = (missing_count / len(df)) * 100
        print(f"  {col:25} | {missing_count:3d} valores ({missing_pct:5.1f}%)")

    # Análise de zeros em colunas críticas
    print(f"\nAnálise de zeros em colunas biológicas:")
    biological_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in biological_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / len(df)) * 100
            print(f"  {col:25} | {zero_count:3d} zeros ({zero_pct:5.1f}%)")

    print(f"\nEstatísticas básicas:")
    print(df.describe().round(2))


def conservative_preprocessing(input_file, output_file):
    """
    Pré-processamento CONSERVADOR - aplica apenas correções essenciais

    Princípios:
    1. Só trata dados claramente inválidos
    2. Mantém a distribuição original
    3. Usa métodos estatisticamente robustos
    4. NÃO remove dados (apenas corrige)
    """

    print("=" * 60)
    print("PRÉ-PROCESSAMENTO CONSERVADOR")
    print("=" * 60)

    # Carregar dados
    print(f"\n1. Carregando dados originais de: {input_file}")
    df = pd.read_csv(input_file)

    # Análise inicial
    analyze_data_quality(df)

    # Criar cópia para processamento
    df_clean = df.copy()
    changes_made = []

    print(f"\n2. Identificando valores biologicamente impossíveis:")

    # ETAPA 1: Tratar apenas zeros BIOLOGICAMENTE IMPOSSÍVEIS
    # Essas são as únicas correções que aplicamos - zeros que são claramente erros
    biological_impossible_zero = {
        "Glucose": "Glicose não pode ser zero em pessoa viva",
        "BloodPressure": "Pressão arterial não pode ser zero em pessoa viva",
        "BMI": "IMC não pode ser zero em pessoa com peso/altura",
    }

    for col, reason in biological_impossible_zero.items():
        if col in df_clean.columns:
            zero_count = (df_clean[col] == 0).sum()
            if zero_count > 0:
                print(f"   • {col}: {zero_count} zeros → NaN ({reason})")
                df_clean[col] = df_clean[col].replace(0, np.nan)
                changes_made.append(f"{col}: {zero_count} zeros convertidos para NaN")

    # ETAPA 2: Tratar zeros QUESTIONÁVEIS mas não impossíveis
    # Para estes, somos mais conservadores
    questionable_zero = {
        "SkinThickness": "Espessura da pele zero é questionável mas possível",
        "Insulin": "Insulina zero é rara mas pode ocorrer em alguns casos",
    }

    print(f"\n   Zeros questionáveis (tratamento conservador):")
    for col, reason in questionable_zero.items():
        if col in df_clean.columns:
            zero_count = (df_clean[col] == 0).sum()
            if zero_count > 0:
                # Só converte se for mais de 10% dos dados (indicativo de erro sistemático)
                zero_pct = (zero_count / len(df_clean)) * 100
                if zero_pct > 10:
                    print(
                        f"   • {col}: {zero_count} zeros → NaN ({zero_pct:.1f}% indica erro sistemático)"
                    )
                    df_clean[col] = df_clean[col].replace(0, np.nan)
                    changes_made.append(
                        f"{col}: {zero_count} zeros convertidos para NaN"
                    )
                else:
                    print(
                        f"   • {col}: {zero_count} zeros mantidos ({zero_pct:.1f}% - pode ser real)"
                    )

    print(f"\n3. Imputação conservadora de dados faltantes:")

    # ETAPA 3: Imputação CONSERVADORA
    # Usamos KNN que preserva relações entre variáveis melhor que mediana/média
    numeric_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    # Verificar se há dados para imputar
    missing_counts = df_clean[numeric_cols].isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing > 0:
        print(f"   Dados faltantes encontrados: {total_missing}")
        print(f"   Método: KNN Imputer (k=5) - preserva relações entre variáveis")

        # KNN Imputer - método mais conservador que preserva padrões dos dados
        imputer = KNNImputer(n_neighbors=5, weights="uniform")

        # Aplicar imputação apenas nas colunas numéricas
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

        for col in numeric_cols:
            if missing_counts[col] > 0:
                print(f"   • {col}: {missing_counts[col]} valores imputados")
                changes_made.append(
                    f"{col}: {missing_counts[col]} valores imputados via KNN"
                )
    else:
        print(f"   Nenhum dado faltante encontrado após conversão de zeros.")

    print(f"\n4. Verificações finais:")

    # Verificar se ainda há NaN
    final_missing = df_clean.isnull().sum().sum()
    if final_missing > 0:
        print(f"   ⚠️  ATENÇÃO: {final_missing} dados ainda faltantes!")
        # Aplicar imputação final com mediana (método mais robusto)
        for col in numeric_cols:
            missing_in_col = df_clean[col].isnull().sum()
            if missing_in_col > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(
                    f"      {col}: {missing_in_col} preenchidos com mediana ({median_val:.2f})"
                )
    else:
        print(f"   ✓ Nenhum dado faltante restante")

    # Verificar tipos de dados
    print(f"   ✓ Verificando tipos de dados...")
    for col in numeric_cols:
        if df_clean[col].dtype == "object":
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            print(f"   • {col}: convertido para numérico")

    # Verificar valores negativos (que seriam biologicamente impossíveis)
    print(f"   ✓ Verificando valores negativos...")
    negative_found = False
    for col in numeric_cols:
        negative_count = (df_clean[col] < 0).sum()
        if negative_count > 0:
            print(f"   ⚠️  {col}: {negative_count} valores negativos encontrados")
            # Converter negativos para valores absolutos (assumindo erro de entrada)
            df_clean[col] = df_clean[col].abs()
            negative_found = True
            changes_made.append(f"{col}: {negative_count} valores negativos corrigidos")

    if not negative_found:
        print(f"   ✓ Nenhum valor negativo encontrado")

    print(f"\n5. Salvando dados pré-processados:")

    # Salvar dados limpos
    df_clean.to_csv(output_file, index=False)
    print(f"   Arquivo salvo: {output_file}")

    # Relatório final
    print(f"\n" + "=" * 60)
    print("RELATÓRIO FINAL DO PRÉ-PROCESSAMENTO CONSERVADOR")
    print("=" * 60)

    print(f"Dataset original:      {df.shape}")
    print(f"Dataset pré-processado: {df_clean.shape}")
    print(f"Linhas removidas:      0 (CONSERVADOR - nenhuma linha removida)")
    print(f"Colunas removidas:     0 (CONSERVADOR - nenhuma coluna removida)")

    print(f"\nMudanças aplicadas:")
    if changes_made:
        for i, change in enumerate(changes_made, 1):
            print(f"  {i}. {change}")
    else:
        print(f"  Nenhuma mudança necessária - dados já estavam limpos!")

    print(f"\nComparação de distribuições (média ± desvio):")
    for col in numeric_cols:
        if col in df.columns:
            orig_mean = df[col].mean()
            orig_std = df[col].std()
            new_mean = df_clean[col].mean()
            new_std = df_clean[col].std()

            print(
                f"  {col:20} | Original: {orig_mean:6.2f}±{orig_std:5.2f} | "
                f"Processado: {new_mean:6.2f}±{new_std:5.2f}"
            )

    print(f"\nPrincípios aplicados:")
    print(f"  ✓ Apenas correções essenciais (zeros biologicamente impossíveis)")
    print(f"  ✓ Preservação da distribuição original dos dados")
    print(f"  ✓ Imputação conservadora (KNN preserva relações)")
    print(f"  ✓ Nenhuma remoção de outliers (podem ser casos reais)")
    print(f"  ✓ Nenhuma transformação/normalização aplicada")
    print(f"  ✓ Formato original mantido para compatibilidade")

    print(f"\n✅ PRÉ-PROCESSAMENTO CONSERVADOR CONCLUÍDO!")
    print(f"   O dataset está limpo mas mantém características originais")
    print(f"   Seu script original funcionará normalmente com '{output_file}'")
    print("=" * 60)

    return df_clean


def process_application_data(app_file, processed_app_file, reference_stats=None):
    """
    Aplica o mesmo pré-processamento conservador aos dados de aplicação
    """

    if not os.path.exists(app_file):
        print(f"\n📋 Arquivo {app_file} não encontrado.")
        print(f"   Quando estiver disponível, execute este script novamente.")
        return None

    print(f"\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO DOS DADOS DE APLICAÇÃO")
    print("=" * 60)

    # Carregar dados de aplicação
    print(f"Carregando: {app_file}")
    df_app = pd.read_csv(app_file)
    print(f"Shape: {df_app.shape}")

    # Aplicar as mesmas correções conservadoras
    df_app_clean = df_app.copy()

    # Mesmo tratamento de zeros biologicamente impossíveis
    biological_impossible_zero = ["Glucose", "BloodPressure", "BMI"]
    questionable_zero = ["SkinThickness", "Insulin"]

    print(f"\nAplicando as mesmas correções conservadoras:")

    for col in biological_impossible_zero:
        if col in df_app_clean.columns:
            zero_count = (df_app_clean[col] == 0).sum()
            if zero_count > 0:
                df_app_clean[col] = df_app_clean[col].replace(0, np.nan)
                print(f"  • {col}: {zero_count} zeros → NaN")

    for col in questionable_zero:
        if col in df_app_clean.columns:
            zero_count = (df_app_clean[col] == 0).sum()
            zero_pct = (zero_count / len(df_app_clean)) * 100
            if zero_count > 0 and zero_pct > 10:
                df_app_clean[col] = df_app_clean[col].replace(0, np.nan)
                print(f"  • {col}: {zero_count} zeros → NaN")

    # Imputação conservadora
    numeric_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    missing_total = df_app_clean[numeric_cols].isnull().sum().sum()
    if missing_total > 0:
        print(f"\nImputando {missing_total} valores faltantes com KNN...")
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        df_app_clean[numeric_cols] = imputer.fit_transform(df_app_clean[numeric_cols])

    # Salvar
    df_app_clean.to_csv(processed_app_file, index=False)
    print(f"\nDados de aplicação pré-processados salvos em: {processed_app_file}")

    return df_app_clean


def main():
    """Função principal"""

    # Arquivos
    INPUT_FILE = "diabetes_dataset.csv"
    OUTPUT_FILE = "diabetes_dataset_clean.csv"  # Nome simples para facilitar uso

    APP_INPUT_FILE = "diabetes_app.csv"
    APP_OUTPUT_FILE = "diabetes_app_clean.csv"

    # Verificar arquivo de entrada
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Erro: Arquivo '{INPUT_FILE}' não encontrado!")
        print(f"   Certifique-se de que o arquivo está no diretório atual.")
        return

    try:
        # Pré-processamento conservador dos dados principais
        df_clean = conservative_preprocessing(INPUT_FILE, OUTPUT_FILE)

        # Pré-processamento dos dados de aplicação (se existirem)
        process_application_data(APP_INPUT_FILE, APP_OUTPUT_FILE)

        print(f"\n🎉 PRONTO PARA USO!")
        print(f"\nPara usar com seu script original:")
        print(
            f"1. Simplesmente substitua 'diabetes_dataset.csv' por 'diabetes_dataset_clean.csv'"
        )
        print(f"2. E 'diabetes_app.csv' por 'diabetes_app_clean.csv' (se aplicável)")
        print(f"3. Todo o resto permanece exatamente igual!")

        print(f"\nAlternativamente, você pode:")
        print(
            f"• Renomear o arquivo original: mv diabetes_dataset.csv diabetes_dataset_original.csv"
        )
        print(f"• Renomear o limpo: mv diabetes_dataset_clean.csv diabetes_dataset.csv")
        print(f"• Usar seu script sem mudanças!")

    except Exception as e:
        print(f"❌ Erro durante pré-processamento: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
