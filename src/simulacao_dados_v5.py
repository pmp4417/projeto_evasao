
import pandas as pd
import numpy as np

def simular_dados(n=500, seed=42):
    np.random.seed(seed)
    
    dados = pd.DataFrame({
        "frequência": np.random.normal(80, 10, n).clip(0, 100),
        "nota_média": np.random.normal(6.5, 2, n).clip(0, 10),
        "participação": np.random.binomial(1, 0.6, n),
        "idade": np.random.normal(24, 5, n).astype(int).clip(16, 45),
        "sexo": np.random.choice(["Masculino", "Feminino"], n),
        "curso": np.random.choice(["Engenharia", "Administração", "Direito", "Psicologia"], n),
        "bolsa": np.random.choice(["Integral", "Parcial", "Nenhuma"], n),
        "turno": np.random.choice(["Manhã", "Tarde", "Noite"], n),
    })
    
    prob_evasao = (
        0.4 * (dados["frequência"] < 70).astype(int) +
        0.3 * (dados["nota_média"] < 5).astype(int) +
        0.2 * (dados["participação"] == 0).astype(int) +
        np.random.rand(n) * 0.2
    )
    
    dados["evasao"] = (prob_evasao > 0.5).astype(int)
    return dados
