import pandas as pd

def aplicar_nivel_risco(model, X, threshold_baixo=0.3, threshold_medio=0.6):
    riscos = model.predict_proba(X)[:, 1]
    def categorizar(prob):
        if prob < threshold_baixo:
            return 'Baixo'
        elif prob < threshold_medio:
            return 'Médio'
        else:
            return 'Alto'
    niveis = [categorizar(p) for p in riscos]
    return pd.Series(riscos), pd.Series(niveis)

def aplicar_intervencoes(df):
    df = df.copy()
    df['frequência'] = df['frequência'] * 1.1
    df['nota_média'] = df['nota_média'] * 1.05
    df['frequência'] = df['frequência'].clip(upper=100)
    df['nota_média'] = df['nota_média'].clip(upper=10)
    return df

def analisar_impacto(original_risco, novo_risco):
    return (original_risco - novo_risco).mean()

def calcular_roi(alunos_afetados, impacto_medio, custo_intervencao, perda_por_aluno=15000):
    beneficio_total = alunos_afetados * impacto_medio * perda_por_aluno
    custo_total = alunos_afetados * custo_intervencao
    return (beneficio_total - custo_total) / custo_total
