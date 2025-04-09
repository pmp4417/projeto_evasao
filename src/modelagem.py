from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def preparar_dados(df):
    df = df.copy()
    le = LabelEncoder()

    # Garantir que a coluna 'evasao' exista
    if 'evasao' not in df.columns:
        if 'evadiu' in df.columns:
            df.rename(columns={'evadiu': 'evasao'}, inplace=True)
        else:
            raise KeyError("A coluna 'evasao' (ou 'evadiu') não foi encontrada no DataFrame.")

    # Codificar colunas categóricas se existirem
    colunas_categoricas = ['sexo', 'curso', 'bolsa', 'turno', 'tipo_ingresso']  # Adicionei 'bolsa' aqui
    for col in colunas_categoricas:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
        else:
            print(f"Aviso: coluna '{col}' não encontrada no DataFrame.")

    # Dividir X e y
    X = df.drop(columns='evasao')
    y = df['evasao']

    return train_test_split(X, y, test_size=0.3, random_state=42)


def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def obter_importancias(model, X_train):
    importances = model.feature_importances_
    return pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)