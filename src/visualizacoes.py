
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st

def plot_risco(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['risco_evasao'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribuição do Risco de Evasão")
    ax.set_xlabel("Risco estimado")
    ax.set_ylabel("Número de alunos")
    st.pyplot(fig)

def plot_boxplot_nivel(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="nivel_risco", y="risco_evasao", data=df, order=["Baixo", "Médio", "Alto"], ax=ax)
    ax.set_title("Risco por Nível de Classificação")
    st.pyplot(fig)

def plot_comparativo_impacto(risco_antes, risco_depois):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(risco_antes, label="Antes da Intervenção", shade=True)
    sns.kdeplot(risco_depois, label="Depois da Intervenção", shade=True)
    ax.set_title("Distribuição do Risco - Antes vs Depois")
    ax.set_xlabel("Risco")
    ax.legend()
    st.pyplot(fig)

def plot_qtd_por_nivel(df):
    fig, ax = plt.subplots()
    sns.countplot(x="nivel_risco", data=df, order=["Baixo", "Médio", "Alto"], ax=ax)
    ax.set_title("Quantidade de Alunos por Nível de Risco")
    st.pyplot(fig)

def plot_pairplot_eda(df):
    st.subheader("🔍 Análise Exploratória")
    st.caption("Abaixo, visualizamos a relação entre variáveis simuladas e evasão.")
    try:
        pairplot_fig = sns.pairplot(df.sample(300), hue='evasao')
        st.pyplot(pairplot_fig)
    except KeyError as e:
        st.error(f"Erro: a coluna '{e.args[0]}' não foi encontrada no DataFrame.")
