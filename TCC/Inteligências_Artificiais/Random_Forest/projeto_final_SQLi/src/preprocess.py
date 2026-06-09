import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from src.feature_extractor import extrair_features 

def processar_dataset(tipo_ataque):
    config = {
        'XSS': {
            'input': 'data/raw/XSS/XSS_dataset.csv',
            'output': 'data/processed/XSS/dataset_distribuido.csv'
        },
        'SQLI': {
            'input': 'data/raw/SQLi/SQLi_dataset.csv',
            'output': 'data/processed/SQLi/dataset_distribuido.csv'
        }
    }

    path_in = config[tipo_ataque]['input']
    path_out = config[tipo_ataque]['output']

    if not os.path.exists(path_in):
        print(f"Erro: Arquivo {path_in} não encontrado.")
        return

    # 1. LER O DATASET RAW
    df_raw = pd.read_csv(path_in)
    
    # Identifica colunas
    coluna_label = next((c for c in df_raw.columns if c.lower() in ['class', 'label']), 'Label')
    coluna_texto = 'Query' if 'Query' in df_raw.columns else df_raw.columns[0]

    # 2. EXTRAÇÃO DE FEATURES (Transforma Texto em Números)
    print(f"[INFO] Extraindo features para {tipo_ataque}...")
    
    features_list = []
    for payload in df_raw[coluna_texto]:
        features_list.append(extrair_features(str(payload)))
    
    df_numerico = pd.DataFrame(features_list)
    df_numerico['label'] = df_raw[coluna_label].values

    # 3. BALANCEAMENTO 70/30 (Individual por tipo para manter qualidade)
    df_normal = df_numerico[df_numerico['label'] == 0]
    df_ataque = df_numerico[df_numerico['label'] == 1]

    # Se você quer 90% normal e 10% ataque:
    n_ataques = len(df_ataque)
    n_normal_proporcional = n_ataques * 9 

    # Garante que não vamos pedir mais do que temos no dataset
    n_amostras_normal = min(len(df_normal), n_normal_proporcional)

    if n_amostras_normal > 0:
        df_normal_reduzido = df_normal.sample(n=n_amostras_normal, random_state=42)
        df_balanceado = pd.concat([df_normal_reduzido, df_ataque]).sample(frac=1, random_state=42)
    else:
        df_balanceado = df_numerico # Fallback se algo der errado

    # 4. NORMALIZAÇÃO
    X = df_balanceado.drop('label', axis=1)
    y = df_balanceado['label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['label'] = y.values

    # 5. SALVAMENTO INDIVIDUAL
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df_final.to_csv(path_out, index=False)
    
    print(f"--- {tipo_ataque} CONCLUÍDO ---")
    return path_out

def unificar_datasets():
    """Lê os CSVs processados de XSS e SQLI e gera um único dataset mestre."""
    print("\n[INFO] Criando Dataset Mestre Unificado...")
    
    path_xss = 'data/processed/XSS/dataset_distribuido.csv'
    path_sqli = 'data/processed/SQLi/dataset_distribuido.csv'
    path_mestre = 'data/processed/dataset_final_mestre.csv'

    if not os.path.exists(path_xss) or not os.path.exists(path_sqli):
        print("Erro: Datasets processados não encontrados para unificação.")
        return

    df_xss = pd.read_csv(path_xss)
    df_sqli = pd.read_csv(path_sqli)

    # Une os dois datasets (concatenação vertical)
    # fillna(0) por segurança, caso os nomes de colunas variem
    df_mestre = pd.concat([df_xss, df_sqli], axis=0).fillna(0)

    # Embaralha tudo para o modelo não viciar na ordem
    df_mestre = df_mestre.sample(frac=1, random_state=42)

    df_mestre.to_csv(path_mestre, index=False)
    print(f"--- SUCESSO: Dataset Mestre salvo em {path_mestre} ---")
    print(f"Total de amostras unificadas: {len(df_mestre)}")

if __name__ == "__main__":
    processar_dataset('XSS')
    processar_dataset('SQLI')
    unificar_datasets()