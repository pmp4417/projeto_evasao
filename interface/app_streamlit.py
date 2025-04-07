import sys
import os

# Adiciona o caminho da raiz do projeto para importar o pacote src
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
from src.simulacao_dados import simular_dados
from src.modelagem import preparar_dados, treinar_modelo
from src.intervencoes import aplicar_nivel_risco, aplicar_intervencoes
from src.visualizacoes import plot_risco
from src.configuracoes import THRESHOLD_RISCO

st.set_page_config(page_title="Previsão de Evasão", layout="centered")
st.title("🎓 Previsão de Evasão Escolar com IA")

threshold = st.slider("Defina o limite de risco para intervenção:", 0.0, 1.0, THRESHOLD_RISCO)

# 1. Simula os dados
df = simular_dados(n=500)

# 2. Prepara os dados e separa treino/teste
X_train, X_test, y_train, y_test = preparar_dados(df)

# 3. Treina o modelo com dados de treino
modelo = treinar_modelo(X_train, y_train)

# 4. Aplica o modelo no conjunto de teste
riscos, niveis = aplicar_nivel_risco(modelo, X_test)

# 5. Adiciona ao DataFrame
df_avaliado = X_test.copy()
df_avaliado["risco_evasao"] = riscos
df_avaliado["nivel_risco"] = niveis

# 6. Aplica intervenções
df_intervencao = aplicar_intervencoes(df_avaliado)

# 7. Interface Streamlit
st.success(f"Intervenções aplicadas com risco ≥ {threshold:.2f}")
plot_risco(df_avaliado)

# Exibe um impacto estimado de forma simbólica (pode ser substituído por análise real)
st.metric("Impacto estimado (placeholder)", f"{threshold * 100:.2f}%")
