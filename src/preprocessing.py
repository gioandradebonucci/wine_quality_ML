"""
preprocessing.py
----------------
Funções auxiliares de pré-processamento para o pipeline de
classificação da qualidade de vinhos.

Uso:
    from src.preprocessing import load_data, apply_log1p, engineer_features, build_feature_matrix
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── Features que recebem transformação log1p ──────────────────────────────────
SKEWED_FEATURES = [
    "residual sugar",
    "chlorides",
    "sulphates",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "fixed acidity",
]

# ── URL fallback do dataset ───────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/gioandradebonucci/"
    "Classificao_da_qualidade_dos_vinhos_usando_ML/refs/heads/main/data/WineQT.csv"
)


def load_data(data_path: str = "../data/WineQT.csv") -> pd.DataFrame:
    """
    Carrega o dataset WineQT com fallback para URL do GitHub.

    Parâmetros
    ----------
    data_path : str
        Caminho local para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo, sem a coluna 'Id' e sem duplicatas.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[load_data] Dataset carregado localmente: {data_path}")
    else:
        df = pd.read_csv(DATASET_URL)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[load_data] Dataset baixado da URL e salvo em: {data_path}")

    # Remove coluna de ID sequencial — sem valor preditivo
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Remove duplicatas exatas
    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    print(f"[load_data] {n_before - len(df)} duplicatas removidas. Shape final: {df.shape}")

    return df


def binarize_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Cria a variável alvo binária quality_label.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna 'quality'.
    threshold : int
        Corte para Alta Qualidade (padrão = 7, conforme enunciado).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas 'quality_label' e 'quality_class' adicionadas.
    """
    df = df.copy()
    df["quality_label"] = (df["quality"] >= threshold).astype(int)
    df["quality_class"] = df["quality_label"].map(
        {0: f"Baixa/Média (< {threshold})", 1: f"Alta (≥ {threshold})"}
    )

    counts = df["quality_label"].value_counts()
    ratio = counts[0] / counts[1]
    print(
        f"[binarize_target] Alta: {counts[1]} ({counts[1]/len(df)*100:.1f}%) | "
        f"Baixa/Média: {counts[0]} ({counts[0]/len(df)*100:.1f}%) | "
        f"Razão: {ratio:.1f}:1"
    )
    return df


def apply_log1p(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Aplica transformação log1p nas features assimétricas.

    Parâmetros
    ----------
    df : pd.DataFrame
    features : list, opcional
        Lista de features a transformar. Se None, usa SKEWED_FEATURES.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame com colunas '_log' adicionadas.
    """
    if features is None:
        features = SKEWED_FEATURES

    df = df.copy()
    for feat in features:
        if feat in df.columns:
            before = df[feat].skew()
            df[feat + "_log"] = np.log1p(df[feat])
            after = df[feat + "_log"].skew()
            reduction = (1 - abs(after) / abs(before)) * 100
            print(f"[log1p] {feat:<30} skew: {before:.3f} → {after:.3f} ({reduction:.1f}% redução)")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features combinadas com racional enológico.

    Features criadas:
    - acid_ratio      : fixed acidity / (volatile acidity + ε)
    - so2_ratio       : free SO2 / (total SO2 + ε)
    - alcohol_density : alcohol / density

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Equilíbrio entre acidez estrutural e acidez acética
    df["acid_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 0.001)

    # Eficiência do conservante SO₂
    df["so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 0.001)

    # Proxy da completude da fermentação
    df["alcohol_density"] = df["alcohol"] / df["density"]

    print("[engineer_features] Features criadas: acid_ratio, so2_ratio, alcohol_density")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    all_original_features: list,
    skewed: list = None,
) -> tuple:
    """
    Monta a matriz de features final (log1p + engineered) e o vetor alvo.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas log1p e engineered já criadas.
    all_original_features : list
        Lista com as 11 features originais.
    skewed : list, opcional
        Features que receberam log1p. Se None, usa SKEWED_FEATURES.

    Retorna
    -------
    tuple : (X, y, features_model)
    """
    if skewed is None:
        skewed = SKEWED_FEATURES

    features_model = [
        (feat + "_log" if feat in skewed else feat) for feat in all_original_features
    ] + ["acid_ratio", "so2_ratio", "alcohol_density"]

    X = df[features_model].copy()
    y = df["quality_label"].copy()

    print(f"[build_feature_matrix] Features no modelo: {len(features_model)}")
    return X, y, features_model


def scale_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Normaliza com RobustScaler e divide em treino/teste estratificado.

    Parâmetros
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    random_state : int

    Retorna
    -------
    tuple : (X_scaled, X_train, X_test, y_train, y_test, scaler)
    """
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"[scale_and_split] Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}"
    )
    return X_scaled, X_train, X_test, y_train, y_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Aplica SMOTE exclusivamente no conjunto de treino.

    Parâmetros
    ----------
    X_train, y_train : dados de treino
    k_neighbors : int
    random_state : int

    Retorna
    -------
    tuple : (X_train_bal, y_train_bal)
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(
        f"[apply_smote] Após SMOTE — Baixa/Média: {(y_train_bal == 0).sum()} | "
        f"Alta: {(y_train_bal == 1).sum()}"
    )
    return X_train_bal, y_train_bal


def run_full_pipeline(data_path: str = "../data/WineQT.csv") -> dict:
    """
    Executa o pipeline completo de pré-processamento.

    Retorna
    -------
    dict com todas as variáveis necessárias para modelagem:
        df, df_proc, X, y, features_model,
        X_scaled, X_train, X_test, y_train, y_test,
        X_train_bal, y_train_bal, scaler
    """
    df = load_data(data_path)
    df = binarize_target(df)

    features = [
        c for c in df.columns
        if c not in ["quality", "quality_label", "quality_class"]
    ]

    df_proc = apply_log1p(df)
    df_proc = engineer_features(df_proc)

    X, y, features_model = build_feature_matrix(df_proc, features)
    X_scaled, X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    return {
        "df": df,
        "df_proc": df_proc,
        "features": features,
        "X": X,
        "y": y,
        "features_model": features_model,
        "X_scaled": X_scaled,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "scaler": scaler,
    }
