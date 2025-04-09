
import sys
import os

# Adiciona o caminho da raiz do projeto para importar o pacote src
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from src.simulacao_dados import simular_dados
from src.modelagem import preparar_dados, treinar_modelo
from src.intervencoes import aplicar_nivel_risco, aplicar_intervencoes, analisar_impacto
from src.visualizacoes import (
    plot_risco,
    plot_boxplot_nivel,
    plot_comparativo_impacto,
    plot_qtd_por_nivel,
)
from src.configuracoes import THRESHOLD_RISCO

st.set_page_config(page_title="Previsão de Evasão", layout="wide")
st.title("🎓 Previsão de Evasão Escolar com IA")

threshold = st.slider("Defina o limite de risco para intervenção:", 0.0, 1.0, THRESHOLD_RISCO)

# Simula os dados
df = simular_dados(n=500)

# Seção de EDA com Pairplot
st.subheader("🔍 Análise Exploratória")
st.markdown("Abaixo, visualizamos a relação entre variáveis simuladas e evasão.")

df_eda = simular_dados(n=300)
pairplot_fig = sns.pairplot(df_eda, hue='evasao')

st.pyplot(pairplot_fig)

# Prepara os dados
X_train, X_test, y_train, y_test = preparar_dados(df)

# Treina o modelo
modelo = treinar_modelo(X_train, y_train)

# Aplica riscos no conjunto de teste
riscos, niveis = aplicar_nivel_risco(modelo, X_test)
df_avaliado = X_test.copy()
df_avaliado["risco_evasao"] = riscos
df_avaliado["nivel_risco"] = niveis

# Exibe gráficos visuais
st.subheader("📊 Distribuições e Perfis de Risco")
plot_risco(df_avaliado)
plot_boxplot_nivel(df_avaliado)
plot_qtd_por_nivel(df_avaliado)

# Exibe tabela de dados
st.subheader("📄 Alunos com risco de evasão")
st.dataframe(df_avaliado)

# Aplica intervenções
df_intervencao = aplicar_intervencoes(df_avaliado)

# Garante que só as features usadas no treino sejam passadas
X_intervencao = df_intervencao[X_train.columns]

# Recalcula riscos após intervenção
riscos_pos, _ = aplicar_nivel_risco(modelo, X_intervencao)

# Calcula impacto médio
impacto = analisar_impacto(df_avaliado["risco_evasao"], riscos_pos)

# Gráfico de impacto comparativo
st.subheader("📉 Comparativo de Risco - Antes vs Depois da Intervenção")
plot_comparativo_impacto(df_avaliado["risco_evasao"], riscos_pos)

st.success(f"Intervenções aplicadas com risco ≥ {threshold:.2f}")
st.metric("Redução média no risco de evasão", f"{impacto:.2%}")

# Botão para baixar os dados
st.download_button("📥 Baixar dados avaliados (CSV)", df_avaliado.to_csv(index=False), "avaliacao.csv", "text/csv")
