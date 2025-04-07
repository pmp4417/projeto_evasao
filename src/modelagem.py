
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def preparar_dados(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ['sexo', 'curso', 'turno', 'tipo_ingresso']:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns='evadiu')
    y = df['evadiu']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def treinar_modelo(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def obter_importancias(model, X_train):
    import pandas as pd
    importances = model.feature_importances_
    return pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
