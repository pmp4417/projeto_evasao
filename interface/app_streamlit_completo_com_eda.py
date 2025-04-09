
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.simulacao_dados import simular_dados
from src.modelagem import treinar_modelo
from src.intervencoes import aplicar_intervencoes
from src.configuracoes import THRESHOLD_RISCO

st.set_page_config(page_title="Previsão de Evasão", layout="centered")
st.title("🎓 Previsão de Evasão Escolar com IA")

threshold = st.slider("Defina o limite de risco para intervenção:", 0.0, 1.0, THRESHOLD_RISCO)

# Simulação de dados
st.info("Simulando dados de alunos...")
df = simular_dados(n=500)

# 🔍 EDA: Visualização pairplot
st.subheader("🔍 Análise Exploratória (EDA)")
st.markdown("Visualização da relação entre frequência, nota, participação e evasão.")
fig = sns.pairplot(df.sample(300), hue='evasao')
st.pyplot(fig)

# Treinamento e classificação
X = df.drop(columns='evasao')
y = df['evasao']
modelo = treinar_modelo(X, y)

df['risco_evasao'] = modelo.predict_proba(X)[:, 1]
df['nivel_risco'] = df['risco_evasao'].apply(
    lambda p: 'Baixo' if p < 0.3 else ('Médio' if p < 0.6 else 'Alto'))

# Aplicar intervenções
df_intervencao = aplicar_intervencoes(df)
df_intervencao['risco_evasao'] = modelo.predict_proba(df_intervencao[X.columns])[:, 1]

# Visualização de comparativo
st.subheader("📉 Comparativo de Risco: Antes vs Depois das Intervenções")
fig2, ax = plt.subplots()
sns.kdeplot(df['risco_evasao'], label='Antes', shade=True, ax=ax)
sns.kdeplot(df_intervencao['risco_evasao'], label='Depois', shade=True, ax=ax)
plt.title("Distribuição de Risco de Evasão")
plt.legend()
st.pyplot(fig2)

# Impacto
impacto = (df['risco_evasao'] - df_intervencao['risco_evasao']).mean()
st.metric("Impacto estimado após intervenção", f"{impacto:.2%}")
