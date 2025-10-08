import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats


def load_and_preprocess(
    file_path,
    age_range=(0, 120),
    remove_outliers=True,
    outlier_method="iqr",
    iqr_multiplier=1.5,
    zscore_threshold=3,
):
    """
    Carrega e preprocessa dados para clustering

    Args:
        file_path: Caminho do arquivo CSV
        age_range: Tupla (min, max) para validar idade
        remove_outliers: Se True, remove outliers
        outlier_method: 'iqr' ou 'zscore'
        iqr_multiplier: Multiplicador para IQR
        zscore_threshold: Limiar para Z-Score

    Returns:
        tuple: (dados_originais, dados_escalados, nomes_features, scaler)
    """

    # Carrega dados
    df = pd.read_csv(file_path)
    print(f"✓ Dados carregados: {df.shape[0]} amostras")

    # Valida e limpa idade
    if "Age" in df.columns:
        invalid_age = (df["Age"] < age_range[0]) | (df["Age"] > age_range[1])
        if invalid_age.sum() > 0:
            print(f"⚠ Removendo {invalid_age.sum()} registros com idade inválida")
            df = df[~invalid_age]

    # Features de espessura epitelial
    epithelial_features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]

    # Remove linhas com valores ausentes
    initial_rows = len(df)
    df_clean = df.dropna(subset=epithelial_features)
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"⚠ Removendo {removed} registros com valores ausentes")

    # Seleciona features
    features_df = df_clean[epithelial_features].copy()

    # Remove outliers se solicitado
    if remove_outliers:
        mask = np.ones(len(features_df), dtype=bool)

        if outlier_method == "iqr":
            for col in epithelial_features:
                Q1 = features_df[col].quantile(0.25)
                Q3 = features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_multiplier * IQR
                upper = Q3 + iqr_multiplier * IQR
                mask &= (features_df[col] >= lower) & (features_df[col] <= upper)

        elif outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(features_df))
            mask = (z_scores < zscore_threshold).all(axis=1)

        outliers_removed = (~mask).sum()
        if outliers_removed > 0:
            print(f"⚠ Removendo {outliers_removed} outliers ({outlier_method})")
            features_df = features_df[mask]
            df_clean = df_clean[mask]

    print(f"✓ Dataset final: {len(features_df)} amostras")

    # Normaliza
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)

    return df_clean, scaled_data, epithelial_features, scaler
