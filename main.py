
from src.simulacao_dados import simular_dados
from src.modelagem import preparar_dados, treinar_modelo, obter_importancias
from src.intervencoes import aplicar_nivel_risco, aplicar_intervencoes, analisar_impacto, calcular_roi

def main():
    print("Iniciando pipeline de previsão de evasão...
")
    
    # 1. Simulação de dados
    df = simular_dados(n=500)
    print(f"Dataset simulado com {df.shape[0]} registros.
")

    # 2. Preparação dos dados e modelagem
    X_train, X_test, y_train, y_test = preparar_dados(df)
    model = treinar_modelo(X_train, y_train)
    print("Modelo treinado com sucesso.
")

    # 3. Importância das variáveis
    importancias = obter_importancias(model, X_train)
    print("Top 5 variáveis mais importantes:")
    print(importancias.head(), "
")

    # 4. Cálculo do risco e intervenções
    riscos, niveis = aplicar_nivel_risco(model, X_test)
    print(f"Distribuição dos níveis de risco:
{niveis.value_counts()}
")

    # 5. Simulação de intervenções
    df_interv = aplicar_intervencoes(df)
    X_train_new, X_test_new, _, _ = preparar_dados(df_interv)
    novos_riscos, _ = aplicar_nivel_risco(model, X_test_new)

    # 6. Análise de impacto e ROI
    impacto = analisar_impacto(riscos, novos_riscos)
    roi = calcular_roi(len(riscos), impacto, custo_intervencao=1000)

    print(f"Impacto médio da intervenção: {impacto:.2%}")
    print(f"ROI estimado da intervenção: {roi:.2%}")

if __name__ == "__main__":
    main()
