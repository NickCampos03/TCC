import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def processar_e_balancear():
    input_path = 'data/raw/XSS/XSS_dataset.csv'
    output_treino = 'data/processed/XSS/dataset_distribuido.csv'

    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado.")
        return

    # 1. LER O DATASET
    df = pd.read_csv(input_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    coluna_label = 'Class' if 'Class' in df.columns else 'Label'
    
    # 2. LÓGICA DE BALANCEAMENTO 70/30
    # Separamos as classes para garantir a proporção exata
    df_normal = df[df[coluna_label] == 0]
    df_ataque = df[df[coluna_label] == 1]
    
    n_ataques = len(df_ataque)
    # Cálculo: Se n_ataques é 30%, quanto é 70%? (n * 0.7 / 0.3)
    n_normal_proporcional = int((n_ataques * 0.8) / 0.2)
    
    print(f"[INFO] Ataques encontrados: {n_ataques}")
    print(f"[INFO] Selecionando {n_normal_proporcional} amostras normais para manter proporção 70/30.")

    # Amostramos os normais para casar com a proporção
    df_normal_reduzido = df_normal.sample(n=min(len(df_normal), n_normal_proporcional), random_state=42)
    
    # Unimos e embaralhamos
    df_balanceado = pd.concat([df_normal_reduzido, df_ataque]).sample(frac=1, random_state=42)

    # 3. NORMALIZAÇÃO
    scaler = MinMaxScaler()
    X = df_balanceado.drop(coluna_label, axis=1)
    y = df_balanceado[coluna_label]
    
    X_scaled = scaler.fit_transform(X)
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['label'] = y.values

    # 4. SALVAMENTO
    os.makedirs(os.path.dirname(output_treino), exist_ok=True)
    df_final.to_csv(output_treino, index=False)
    
    print(f"--- SUCESSO ---")
    print(f"Total Final: {len(df_final)} (70% Normal / 30% Ataque)")
    print(f"Arquivo atualizado em: {output_treino}")

if __name__ == "__main__":
    processar_e_balancear()