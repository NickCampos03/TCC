import pandas as pd
import os
import urllib.parse

SQLI_RAW_PATH = 'data/SQLi/sqli_raw.csv'  
XSS_RAW_PATH = 'data/XSS/xss_raw.csv'    
OUTPUT_PATH = 'data/dataset_hibrido.csv'

def limpar_e_normalizar(payload):
    """Decodifica e limpa strings para evitar bypasses simples."""
    texto = str(payload).lower()
    # Decodifica URL encoding (Ex: %3C vira <)
    texto = urllib.parse.unquote(texto)
    return texto

def processar_e_unificar():
    print("--- Iniciando Processamento Baseado em Texto (NLP) ---")
    
    if not os.path.exists(SQLI_RAW_PATH):
        print(f"Erro: Arquivo {SQLI_RAW_PATH} não encontrado.")
        return
    
    df_sqli_raw = pd.read_csv(SQLI_RAW_PATH)
    
    # Padroniza nomes de colunas do SQLi
    df_sqli_raw.columns = df_sqli_raw.columns.str.lower()
    coluna_query = 'query' if 'query' in df_sqli_raw.columns else df_sqli_raw.columns[0]
    coluna_label = 'label' if 'label' in df_sqli_raw.columns else df_sqli_raw.columns[1]
    
    # Cria dataframe base de SQLi apenas com texto limpo e label
    print(f"Processando {len(df_sqli_raw)} linhas de SQLi...")
    sqli_dados = []
    for _, row in df_sqli_raw.iterrows():
        texto_limpo = limpar_e_normalizar(row[coluna_query])
        try:
            lbl = int(row[coluna_label])
        except Exception:
            lbl = row[coluna_label]
        sqli_dados.append({'payload': texto_limpo, 'label': lbl})
        
    df_sqli_final = pd.DataFrame(sqli_dados)
    
    # Processa e unifica com o XSS
    if os.path.exists(XSS_RAW_PATH):
        print("Unificando com o dataset de XSS...")
        df_xss_raw = pd.read_csv(XSS_RAW_PATH)
        df_xss_raw.columns = df_xss_raw.columns.str.lower()
        
        # O dataset original do XSS tem 67 colunas. Mas precisamos do texto bruto original dele.
        # Caso o xss_raw.csv já seja o processador de 67 colunas e não tenha o texto original,
        # usaremos uma abordagem adaptativa. Vamos assumir que buscamos uma coluna de texto.
        # Se não houver, reconstruiremos uma string sintética a partir das colunas marcadas com 1.
        xss_dados = []
        
        if 'payload' in df_xss_raw.columns or 'text' in df_xss_raw.columns:
            col_txt = 'payload' if 'payload' in df_xss_raw.columns else 'text'
            col_lbl = 'label' if 'label' in df_xss_raw.columns else ('class' if 'class' in df_xss_raw.columns else df_xss_raw.columns[-1])
            for _, row in df_xss_raw.iterrows():
                xss_dados.append({'payload': limpar_e_normalizar(row[col_txt]), 'label': int(row[col_lbl])})
        else:
            # Se o xss_raw for o dataset numérico puro, recriamos um texto com o nome das colunas ativas
            print("Aviso: Convertendo colunas numéricas de XSS em representação textual para o TF-IDF...")
            col_lbl = 'class' if 'class' in df_xss_raw.columns else ('label' if 'label' in df_xss_raw.columns else df_xss_raw.columns[-1])
            colunas_features = [c for c in df_xss_raw.columns if c != col_lbl]
            
            for _, row in df_xss_raw.iterrows():
                tags_ativas = [c.replace('_', ' ') for c in colunas_features if row[c] > 0]
                texto_sintetizado = " ".join(tags_ativas)
                xss_dados.append({'payload': texto_sintetizado, 'label': int(row[col_lbl])})
                
        df_xss_final = pd.DataFrame(xss_dados)
        
        # Concatena os dois datasets baseados em texto
        df_completo = pd.concat([df_xss_final, df_sqli_final], ignore_index=True)
        df_completo = df_completo.dropna().sample(frac=1).reset_index(drop=True)
        
        df_completo.to_csv(OUTPUT_PATH, index=False)
        print(f"Sucesso! Dataset híbrido de texto salvo em: {OUTPUT_PATH}")
        print(f"Total de registros: {len(df_completo)}")
    else:
        df_sqli_final.dropna().to_csv(OUTPUT_PATH, index=False)
        print(f"Aviso: XSS não encontrado. Apenas SQLi processado salvo em {OUTPUT_PATH}")

if __name__ == "__main__":
    processar_e_unificar()