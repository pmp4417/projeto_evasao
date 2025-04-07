
import numpy as np
import pandas as pd

def simular_dados(n=500, seed=42):
    np.random.seed(seed)
    
    idade = np.random.randint(17, 60, size=n)
    sexo = np.random.choice(['Feminino', 'Masculino'], size=n)
    curso = np.random.choice(['Administração', 'Engenharia', 'Pedagogia', 'Direito', 'Enfermagem'], size=n)
    turno = np.random.choice(['Matutino', 'Vespertino', 'Noturno'], size=n)
    semestre_atual = np.random.randint(1, 10, size=n)
    nota_média = np.round(np.random.normal(6.5, 1.5, size=n), 2)
    frequência = np.round(np.random.normal(85, 10, size=n), 2)
    trancamentos = np.random.poisson(0.5, size=n)
    atraso_pagamento = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    tipo_ingresso = np.random.choice(['Vestibular', 'ENEM', 'Transferência', 'Segunda Graduação'], size=n)
    trabalha = np.random.choice([0, 1], size=n)
    auxílio_bolsa = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    
    evadiu = ((nota_média < 5.5) | 
              (frequência < 70) | 
              (trancamentos > 2) | 
              (atraso_pagamento == 1)).astype(int)

    data = pd.DataFrame({
        'idade': idade,
        'sexo': sexo,
        'curso': curso,
        'turno': turno,
        'semestre_atual': semestre_atual,
        'nota_média': nota_média,
        'frequência': frequência,
        'trancamentos': trancamentos,
        'atraso_pagamento': atraso_pagamento,
        'tipo_ingresso': tipo_ingresso,
        'trabalha': trabalha,
        'auxílio_bolsa': auxílio_bolsa,
        'evadiu': evadiu
    })
    
    return data
