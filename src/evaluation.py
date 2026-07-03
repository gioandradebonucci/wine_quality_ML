"""
evaluation.py
-------------
Funções auxiliares para avaliação e comparação dos modelos de classificação.

Uso:
    from src.evaluation import evaluate_model, compare_models, find_optimal_threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance


# ── Paleta de cores padronizada ───────────────────────────────────────────────
BG_FIG  = "#ffffff"
BG_AX   = "#ffffff"
TEXT    = "#222222"
C_SPINE = "#aaaaaa"
C_LR    = "#800000"
C_RF    = "#003300"
C_GB    = "#8B4513"

COLORS = {
    "Logistic Regression": C_LR,
    "Random Forest":       C_RF,
    "Gradient Boosting":   C_GB,
}


def style_ax(ax):
    """Aplica estilo visual padrão a um eixo matplotlib."""
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=TEXT, labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(C_SPINE)


def evaluate_model(nome: str, modelo, X_test, y_test) -> dict:
    """
    Avalia um modelo no conjunto de teste e retorna todas as métricas.

    Parâmetros
    ----------
    nome : str
        Nome do modelo (para exibição).
    modelo : estimador scikit-learn treinado
    X_test, y_test : dados de teste

    Retorna
    -------
    dict com métricas e predições.
    """
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "PR-AUC":    average_precision_score(y_test, y_prob),
        "Recall":    recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

    print("=" * 60)
    print(f"  {nome}")
    print("=" * 60)
    print(f"  F1-Score:  {metricas['F1-Score']:.4f}  ← MÉTRICA PRINCIPAL")
    print(f"  PR-AUC:    {metricas['PR-AUC']:.4f}  ← mais honesta com desbalanceamento")
    print(f"  ROC-AUC:   {metricas['ROC-AUC']:.4f}")
    print(f"  Recall:    {metricas['Recall']:.4f}")
    print(f"  Precision: {metricas['Precision']:.4f}")
    print(f"  Accuracy:  {metricas['Accuracy']:.4f}  (enganosa com desbalanceamento)")
    print()
    print(classification_report(y_test, y_pred, target_names=["Baixa/Média (0)", "Alta (1)"]))

    return metricas


def compare_models(resultados: dict, y_test, save_path: str = None):
    """
    Gera gráfico comparativo de métricas entre todos os modelos.

    Parâmetros
    ----------
    resultados : dict
        {nome_modelo: dict_de_metricas}
    y_test : array-like
    save_path : str, opcional
        Caminho para salvar o gráfico.
    """
    nomes    = list(resultados.keys())
    cores    = [COLORS.get(n, "#555555") for n in nomes]
    metricas = ["F1-Score", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Accuracy"]
    x        = np.arange(len(metricas))
    width    = 0.22
    baseline = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG_FIG)
    style_ax(ax)

    for i, (nome, cor) in enumerate(zip(nomes, cores)):
        valores = [resultados[nome][m] for m in metricas]
        bars    = ax.bar(
            x + i * width, valores, width,
            label=nome, color=cor, alpha=0.85, edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=11, fontweight="bold", color=TEXT, padding=2)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metricas, fontsize=13, color=TEXT)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Valor da Métrica", fontsize=13, color=TEXT)
    ax.set_title(
        "Comparação de Métricas — Conjunto de Teste (distribuição real)",
        fontsize=14, fontweight="bold", color=TEXT, pad=12,
    )
    ax.legend(facecolor="#eeeeee", labelcolor=TEXT, fontsize=12)
    ax.axhline(baseline, color="#cc6600", linewidth=1.5, linestyle=":")
    ax.text(
        len(metricas) - 0.3, baseline + 0.025,
        f"Baseline PR-AUC = {baseline:.2f}",
        color="#cc6600", fontsize=9, fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.show()


def plot_confusion_matrices(resultados: dict, y_test, save_path: str = None):
    """
    Plota matrizes de confusão para todos os modelos.

    Parâmetros
    ----------
    resultados : dict
    y_test : array-like
    save_path : str, opcional
    """
    nomes   = list(resultados.keys())
    cores   = [COLORS.get(n, "#555555") for n in nomes]
    rotulos = [["VN", "FP"], ["FN", "VP"]]

    fig, axes = plt.subplots(1, len(nomes), figsize=(17, 6), facecolor=BG_FIG)

    for ax, nome, cor in zip(axes, nomes, cores):
        cm  = confusion_matrix(y_test, resultados[nome]["y_pred"])
        pct = cm / cm.sum() * 100
        sns.heatmap(
            cm, annot=False, ax=ax,
            cmap=sns.light_palette(cor, as_cmap=True),
            linewidths=2, linecolor="white", cbar=False,
        )
        threshold = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                cor_t = "white" if cm[i, j] > threshold else "#222222"
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{rotulos[i][j]}\n{cm[i,j]}\n({pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color=cor_t,
                )
        ax.set_xticklabels(["Previu\nBaixa/Média", "Previu\nAlta"], color=TEXT, fontsize=12)
        ax.set_yticklabels(["Real\nBaixa/Média", "Real\nAlta"],     color=TEXT, fontsize=12, rotation=0)
        ax.set_title(nome, fontsize=13, fontweight="bold", color=TEXT, pad=10)

    plt.suptitle(
        "Matrizes de Confusão — Conjunto de Teste\n"
        "(VN/VP = acertos | FP = alarme falso | FN = miss — mais custoso)",
        fontsize=13, fontweight="bold", color=TEXT, y=1.03,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.show()


def find_optimal_threshold(
    y_test,
    y_prob,
    model_name: str = "Gradient Boosting",
    save_path: str = None,
) -> float:
    """
    Encontra o threshold que maximiza o F1-Score.

    Parâmetros
    ----------
    y_test : array-like
    y_prob : array-like
        Probabilidades preditas da classe positiva.
    model_name : str
    save_path : str, opcional

    Retorna
    -------
    float : threshold ótimo
    """
    thresholds  = np.arange(0.10, 0.91, 0.01)
    preds       = [(y_prob >= t).astype(int) for t in thresholds]
    f1_list     = [f1_score(y_test, p, zero_division=0)        for p in preds]
    prec_list   = [precision_score(y_test, p, zero_division=0) for p in preds]
    rec_list    = [recall_score(y_test, p,   zero_division=0)  for p in preds]

    best_idx    = int(np.argmax(f1_list))
    best_thresh = float(thresholds[best_idx])
    best_f1     = float(f1_list[best_idx])
    best_prec   = float(prec_list[best_idx])
    best_rec    = float(rec_list[best_idx])
    f1_padrao   = float(f1_score(y_test, (y_prob >= 0.5).astype(int)))

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_FIG)
    style_ax(ax)
    ax.plot(thresholds, f1_list,   color=C_GB,  linewidth=2.5, label="F1-Score")
    ax.plot(thresholds, prec_list, color="#800000", linewidth=1.8, linestyle="--", label="Precision")
    ax.plot(thresholds, rec_list,  color="#003300", linewidth=1.8, linestyle="--", label="Recall")
    ax.axvline(0.5,         color="#888888", linewidth=1.2, linestyle=":", label="Threshold padrão (0.50)")
    ax.axvline(best_thresh, color=C_GB,      linewidth=2.0, linestyle="-.", label=f"Melhor F1 (threshold={best_thresh:.2f})")
    ax.scatter([best_thresh], [best_f1], color=C_GB, zorder=5, s=80)
    ax.annotate(
        f"F1={best_f1:.3f}\nP={best_prec:.3f}\nR={best_rec:.3f}",
        xy=(best_thresh, best_f1),
        xytext=(best_thresh + 0.06, best_f1 - 0.10),
        fontsize=11, color=TEXT,
        arrowprops=dict(arrowstyle="->", color=C_SPINE),
    )
    ax.set_xlabel("Threshold de decisão", fontsize=13, color=TEXT)
    ax.set_ylabel("Valor da métrica",     fontsize=13, color=TEXT)
    ax.set_title(
        f"{model_name} — F1, Precision e Recall por Threshold\n(threshold ótimo = máximo F1)",
        fontsize=14, fontweight="bold", color=TEXT,
    )
    ax.legend(fontsize=11, facecolor="#eeeeee", labelcolor=TEXT)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.show()

    print("=" * 60)
    print(f"  Threshold padrão  (0.50): F1 = {f1_padrao:.4f}")
    print(f"  Threshold ótimo ({best_thresh:.2f}): F1 = {best_f1:.4f}  (+{best_f1 - f1_padrao:.4f})")
    print(f"  Precision: {best_prec:.4f} | Recall: {best_rec:.4f}")
    print(f"  → Use threshold {best_thresh:.2f} em produção para maximizar F1.")
    print("=" * 60)

    return best_thresh


def plot_feature_importance(
    modelo,
    X_test,
    y_test,
    features_model: list,
    model_name: str = "Modelo",
    save_path: str = None,
):
    """
    Plota Gini Importance e Permutation Importance lado a lado.

    Parâmetros
    ----------
    modelo : estimador treinado (Random Forest ou Gradient Boosting)
    X_test, y_test : dados de teste
    features_model : list
    model_name : str
    save_path : str, opcional
    """
    C_LOW  = "#800000"
    C_HIGH = "#003300"

    imp_gini = pd.Series(
        modelo.feature_importances_, index=features_model
    ).sort_values(ascending=True)

    pi       = permutation_importance(
        modelo, X_test, y_test, n_repeats=15, random_state=42, scoring="f1"
    )
    imp_perm = pd.Series(
        pi.importances_mean, index=features_model
    ).sort_values(ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG_FIG)

    for ax, imp, title in [
        (ax1, imp_gini, f"{model_name}\nGini Importance (redução de impureza)"),
        (ax2, imp_perm, f"{model_name}\nPermutation Importance (impacto no F1)"),
    ]:
        style_ax(ax)
        cores_imp = [C_HIGH if i >= len(imp) - 5 else C_LOW for i in range(len(imp))]
        bars = ax.barh(imp.index, imp.values, color=cores_imp, edgecolor="white")
        ax.bar_label(
            bars, fmt="%.3f", label_type="center",
            color="white", fontsize=11, fontweight="bold", padding=0,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.4, edgecolor="none"),
        )
        ax.set_xlabel("Importância", fontsize=13, color=TEXT)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=10)
        ax.axvline(0, color=C_SPINE, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.show()
