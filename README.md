# 🎓 Previsão de Evasão Escolar com IA

Este projeto é um MVP desenvolvido para a Cruzeiro do Sul Educacional com o objetivo de prever a evasão de alunos em cursos de graduação, utilizando técnicas de Machine Learning e visualizações interativas.

---

## 🚀 Funcionalidades

- Simulação de dados estudantis realistas
- Treinamento de modelo preditivo (Random Forest)
- Classificação dos alunos por nível de risco de evasão (Baixo, Médio, Alto)
- Aplicação de intervenções simuladas (ajuste de frequência e notas)
- Visualização de impacto antes e depois das intervenções
- Avaliação do modelo com métricas e matriz de confusão
- Visualização interativa em Streamlit e Jupyter Notebook

---

## 📊 Tecnologias Utilizadas

- Python
- Streamlit
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib

---

## 🛠️ Como Executar

### Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/pmp4417/projeto_evasao.git
cd projeto_evasao
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executar via Streamlit na raiz do projeto
```bash
streamlit run interface/app_streamlit_final.py
```

### Executar via Jupyter Notebook
Abra o arquivo:
```
MVP_analise_evasao_escolar_com_scroll.ipynb
```

---

## 📂 Estrutura

```
projeto-evasao/
│
├── interface/                  # Apps com Streamlit
├── notebook/                   # Jupyter Notebooks
├── src/                        # Módulos de simulação, modelo e visualização
├── requirements.txt            # Pacotes necessários
├── README.md                   # Este arquivo
```

---

## 📈 Avaliação do Modelo

- **Acurácia:** 66%
- **Recall (evasão):** 78%
- **Erro Tipo II (falso negativo):** minimizado com priorização estratégica

---

## 🤝 Contribuição

Pull requests são bem-vindos. Para mudanças maiores, abra uma issue antes.

---

## 🧠 Autor

Pedro Miguel Pereira 


