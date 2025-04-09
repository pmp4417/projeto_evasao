
import numpy as np
import pandas as pd

def simular_dados(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "frequência": np.random.normal(80, 10, n).clip(40, 100),
        "nota_média": np.random.normal(6, 2, n).clip(0, 10),
        "participação": np.random.choice([0, 1], n)
    })

    # Rótulo de evasão: risco maior se frequência < 70 ou nota < 5
    df["evasao"] = ((df["frequência"] < 70) | (df["nota_média"] < 5)).astype(int)
    return df
