import sys
import os

# Adiciona o caminho da raiz do projeto para importar o pacote src
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
from src.simulacao_dados import simular_dados
from src.modelagem import preparar_dados, treinar_modelo
from src.intervencoes import aplicar_nivel_risco, aplicar_intervencoes, analisar_impacto
from src.visualizacoes import plot_risco
from src.configuracoes import THRESHOLD_RISCO

st.set_page_config(page_title="Previsão de Evasão", layout="wide")
st.title("🎓 Previsão de Evasão Escolar com IA")

threshold = st.slider("Defina o limite de risco para intervenção:", 0.0, 1.0, THRESHOLD_RISCO)

# Simula os dados
df = simular_dados(n=500)

# Prepara os dados
X_train, X_test, y_train, y_test = preparar_dados(df)

# Treina o modelo
modelo = treinar_modelo(X_train, y_train)

# Aplica riscos no conjunto de teste
riscos, niveis = aplicar_nivel_risco(modelo, X_test)
df_avaliado = X_test.copy()
df_avaliado["risco_evasao"] = riscos
df_avaliado["nivel_risco"] = niveis

# Exibe gráfico
st.subheader("📊 Distribuição de risco de evasão")
plot_risco(df_avaliado)

# Exibe tabela de dados
st.subheader("📄 Alunos com risco de evasão")
st.dataframe(df_avaliado)

# Aplica intervenções
df_intervencao = aplicar_intervencoes(df_avaliado)

# Recalcula riscos após intervenção
X_intervencao = df_intervencao[X_train.columns]  # Garante que só passa as features
riscos_pos, _ = aplicar_nivel_risco(modelo, X_intervencao)


# Calcula impacto médio
impacto = analisar_impacto(df_avaliado["risco_evasao"], riscos_pos)

st.success(f"Intervenções aplicadas com risco ≥ {threshold:.2f}")
st.metric("Redução média no risco de evasão", f"{impacto:.2%}")

# Botão para baixar os dados
st.download_button("📥 Baixar dados avaliados (CSV)", df_avaliado.to_csv(index=False), "avaliacao.csv", "text/csv")
