
import pandas as pd
import numpy as np

def simular_dados(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "frequência": np.random.normal(loc=80, scale=10, size=n).clip(40, 100),
        "nota_média": np.random.normal(loc=6.5, scale=2, size=n).clip(0, 10),
        "participação": np.random.choice([0, 1], size=n),
        "idade": np.random.randint(17, 45, size=n),
        "sexo": np.random.choice(["Masculino", "Feminino"], size=n),
        "curso": np.random.choice(["Engenharia", "Administração", "Direito", "Pedagogia"], size=n),
        "bolsa": np.random.choice(["Nenhuma", "Parcial", "Integral"], size=n),
        "turno": np.random.choice(["Manhã", "Tarde", "Noite"], size=n),
        "tipo_ingresso": np.random.choice(["Vestibular", "ENEM", "Transferência", "Outro"], size=n),
    })

    prob_evasao = (
        (100 - df["frequência"]) * 0.02 +
        (6.5 - df["nota_média"]) * 0.05 +
        (1 - df["participação"]) * 0.2 +
        (df["bolsa"] == "Nenhuma").astype(int) * 0.1
    )
    df["evasao"] = (np.random.rand(n) < prob_evasao.clip(0, 1)).astype(int)
    return df
