<<<<<<< HEAD
# 🍷 Classificação da Qualidade de Vinhos com Machine Learning

**FIAP — POSTECH | Tech Challenge Fase 2**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Descrição do Projeto

Pipeline completo de Machine Learning para classificação binária da qualidade de vinhos tintos, desenvolvido como Tech Challenge da Fase 2 do programa POSTECH da FIAP.

O modelo prevê se um vinho é de **Alta Qualidade** (nota ≥ 7) ou **Baixa/Média Qualidade** (nota < 7) com base em suas características físico-químicas mensuradas durante a produção — sem depender de avaliação sensorial humana.

**Problema de negócio:** a avaliação sensorial de vinhos é subjetiva, lenta e cara. Um modelo preditivo permite antecipar a qualidade durante o processo produtivo, reduzindo rejeições e apoiando decisões de precificação.

---

## 🎯 Resultados Principais

| Modelo | F1-Score | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0,488 | 0,469 | 0,886 | 0,741 | 0,364 |
| Random Forest | 0,606 | 0,664 | 0,907 | 0,741 | 0,513 |
| **Gradient Boosting** ✅ | **0,646** | 0,597 | **0,907** | **0,778** | **0,553** |

**Modelo vencedor:** Gradient Boosting com threshold ótimo de **0,55** (F1 = 0,667).

**Features mais importantes:** `alcohol`, `alcohol_density`, `volatile acidity`, `sulphates_log`.

---

## 📁 Estrutura do Repositório

```
wine-quality-classification/
│
├── data/                          # Base de dados
│   └── WineQT.csv                 # Dataset original (baixar — instruções abaixo)
│
├── notebooks/                     # Análise e modelagem
│   └── wine_quality_analysis.ipynb
│
├── src/                           # Scripts auxiliares
│   ├── preprocessing.py           # Funções de pré-processamento
│   └── evaluation.py              # Funções de avaliação dos modelos
│
├── results/                       # Gráficos gerados pelo notebook
│   ├── 01_variavel_alvo.png
│   ├── 02_distribuicao_features.png
│   ├── 03_boxplots_por_classe.png
│   ├── 04_mapa_correlacao.png
│   ├── 05_correlacao_quality.png
│   ├── 05b_pearson_vs_spearman.png
│   ├── 06_comparacao_metricas.png
│   ├── 07_curvas_roc_pr.png
│   ├── 08_matrizes_confusao.png
│   ├── 09_feature_importance.png
│   └── 10_threshold_tuning.png
│
├── docs/                          # Documentação e relatório
│   └── Relatorio_Executivo_Wine_Quality.pdf
│
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

---

## 📦 Dataset

**Fonte:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality)

**Download do arquivo:**
1. Acesse o link acima e baixe `WineQT.csv`
2. Coloque o arquivo na pasta `data/`
3. O notebook carrega automaticamente de `../data/WineQT.csv`

> O notebook também possui fallback automático: se o arquivo não existir localmente, ele baixa do repositório GitHub e salva em `data/` para uso futuro.

**Sobre o dataset:**
- 1.143 amostras originais de vinho tinto português (Vinho Verde)
- 125 duplicatas removidas → **1.018 amostras únicas**
- 11 features físico-químicas + 1 variável de qualidade (escala 3–8)
- Desbalanceamento: 86,5% Baixa/Média × 13,5% Alta

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/gioandradebonucci/Classificao_da_qualidade_dos_vinhos_usando_ML.git
cd Classificao_da_qualidade_dos_vinhos_usando_ML
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset
Coloque `WineQT.csv` na pasta `data/` (instruções acima).

### 5. Execute o notebook
```bash
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

Execute todas as células em sequência. Os gráficos serão salvos automaticamente em `results/`.

---

## 🔬 Pipeline do Projeto

```
1. Compreensão do Problema
   └── Contexto de negócio, glossário das features, binarização da variável alvo

2. Análise Exploratória (EDA)
   ├── Distribuição das features (histograma + KDE + skewness)
   ├── Boxplots por classe (discriminação visual)
   ├── Correlação Pearson vs Spearman (detecção de não-linearidades)
   └── Teste Mann-Whitney U (validação estatística)

3. Pré-processamento
   ├── Transformação log1p (6 features assimétricas)
   ├── Feature Engineering (acid_ratio, so2_ratio, alcohol_density)
   ├── RobustScaler (normalização resistente a outliers)
   ├── Split 80/20 estratificado
   └── SMOTE via ImbPipeline (sem data leakage)

4. Modelagem
   ├── Logistic Regression (baseline linear)
   ├── Random Forest (ensemble bagging)
   └── Gradient Boosting (ensemble boosting)

5. Avaliação
   ├── Validação cruzada estratificada 5-fold
   ├── Métricas: F1, PR-AUC, ROC-AUC, Recall, Precision, Accuracy
   ├── Curvas ROC e Precision-Recall
   ├── Matrizes de confusão
   ├── Feature Importance (Gini + Permutation)
   └── Threshold Tuning (limiar ótimo = 0,55)

6. Interpretação
   └── Implicações práticas para o processo produtivo
```

---

## 📊 Decisões Metodológicas

| Decisão | Alternativa | Por que esta escolha |
|---|---|---|
| RobustScaler | StandardScaler | Resistente a outliers presentes nas features assimétricas |
| SMOTE via ImbPipeline | Random Oversampling | Evita data leakage; gera amostras diversas |
| Mann-Whitney U | t-test | Não assume normalidade — features assimétricas violam o pressuposto |
| PR-AUC como métrica principal | Accuracy | Mais honesta com desbalanceamento 6:1 |
| Threshold 0,55 | Threshold 0,50 | Eleva F1 de 0,646 → 0,667 com Recall mantido em 0,778 |

---

## 📄 Relatório Executivo

O relatório completo com storytelling da análise, interpretação de negócio e roadmap de implementação está disponível em:

📄 [`docs/Relatorio_Executivo_Wine_Quality.pdf`](docs/Relatorio_Executivo_Wine_Quality.pdf)

---

## 🏆 Conclusões

1. **Teor alcoólico** é o principal preditor positivo — controlável via fermentação
2. **Acidez volátil** é o principal preditor negativo — exige controle microbiológico
3. **Feature engineering** gerou ganhos concretos: `alcohol_density` foi a 2ª feature mais importante no Random Forest
4. O problema tem componentes **não-lineares** relevantes — justifica RF e GB sobre LR
5. Com threshold 0,55, o Gradient Boosting atinge F1 = 0,667 e Recall = 0,778

---

## 👥 Autores

Desenvolvido como Tech Challenge Fase 2 — POSTECH FIAP 2026.

---

## 📚 Referências

- CORTEZ, P. et al. Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, v. 47, n. 4, p. 547-553, 2009.
- CHAWLA, N. V. et al. SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, v. 16, p. 321-357, 2002.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825-2830, 2011.
- FRIEDMAN, J. H. Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, v. 29, n. 5, p. 1189-1232, 2001.
=======
# wine_quality_ML
>>>>>>> d465d670344247d0186d5f2fa4e7345ca78465d1
