from dotenv import load_dotenv
import os

load_dotenv()

N_SIMULACOES = int(os.getenv("N_SIMULACOES", 500))
THRESHOLD_RISCO = float(os.getenv("THRESHOLD_RISCO", 0.65))
CAMINHO_SAIDA = os.getenv("CAMINHO_SAIDA", "resultados/")
