
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
from src.simulacao_dados import simular_dados
from src.modelagem import preparar_dados, treinar_modelo
from src.intervencoes import aplicar_nivel_risco, aplicar_intervencoes, analisar_impacto
from src.visualizacoes import (
    plot_risco,
    plot_boxplot_nivel,
    plot_comparativo_impacto,
    plot_qtd_por_nivel,
    plot_pairplot_eda
)
from src.configuracoes import THRESHOLD_RISCO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


st.set_page_config(page_title="Previsão de Evasão", layout="wide")
st.title("🎓 Previsão de Evasão Escolar com IA")

threshold = st.slider("Defina o limite de risco para intervenção:", 0.0, 1.0, THRESHOLD_RISCO)

df = simular_dados(n=500)

# EDA
st.subheader("🔍 Análise Exploratória")
plot_pairplot_eda(df)

X_train, X_test, y_train, y_test = preparar_dados(df)
modelo = treinar_modelo(X_train, y_train)

riscos, niveis = aplicar_nivel_risco(modelo, X_test)
df_avaliado = X_test.copy()
df_avaliado["risco_evasao"] = riscos
df_avaliado["nivel_risco"] = niveis

st.subheader("📊 Distribuições e Perfis de Risco")
plot_risco(df_avaliado)
plot_boxplot_nivel(df_avaliado)
plot_qtd_por_nivel(df_avaliado)

st.subheader("📄 Alunos com risco de evasão")
st.dataframe(df_avaliado)

df_intervencao = aplicar_intervencoes(df_avaliado)
X_intervencao = df_intervencao[X_train.columns]
riscos_pos, _ = aplicar_nivel_risco(modelo, X_intervencao)

impacto = analisar_impacto(df_avaliado["risco_evasao"], riscos_pos)
st.subheader("📉 Comparativo de Risco - Antes vs Depois da Intervenção")
plot_comparativo_impacto(df_avaliado["risco_evasao"], riscos_pos)

st.success(f"Intervenções aplicadas com risco ≥ {threshold:.2f}")
st.metric("Redução média no risco de evasão", f"{impacto:.2%}")
st.download_button("📥 Baixar dados avaliados (CSV)", df_avaliado.to_csv(index=False), "avaliacao.csv", "text/csv")

# 📊 Avaliação do Modelo

st.header("📊 Avaliação do Modelo de Evasão Escolar")

# Treinamento do modelo com os mesmos dados simulados
df_avaliacao = simular_dados(n=500)
X = df_avaliacao.drop(columns='evasao')
y = df_avaliacao['evasao']

# Encoding das variáveis categóricas
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Permanece", "Evade"], yticklabels=["Permanece", "Evade"], ax=ax_cm)
ax_cm.set_xlabel("Predição")
ax_cm.set_ylabel("Real")
ax_cm.set_title("Matriz de Confusão")
st.pyplot(fig_cm)

# Relatório
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.subheader("📋 Relatório de Classificação")
st.dataframe(report_df)

# Importância das variáveis
importancias = modelo.feature_importances_
features = X.columns
importancia_df = pd.DataFrame({'Feature': features, 'Importance': importancias}).sort_values(by='Importance', ascending=False).head(10)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
sns.barplot(data=importancia_df, y='Feature', x='Importance', ax=ax_imp)
ax_imp.set_title('Top 10 Variáveis mais Relevantes')
st.pyplot(fig_imp)

st.info("🔍 Preferimos minimizar o Erro Tipo II (falso negativo), ou seja, evitar deixar um aluno que realmente vai sair sem nenhuma ação preventiva.")
