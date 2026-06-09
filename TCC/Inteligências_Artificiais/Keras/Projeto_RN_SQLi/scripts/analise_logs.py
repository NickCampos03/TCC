import pandas as pd
import numpy as np
import os
import re
import joblib
import tensorflow as tf
import urllib.parse
import json

def limpar_e_normalizar_log(payload):
    """Decodifica, remove a estrutura JSON se existir e normaliza o texto bruto."""
    texto = str(payload).strip()
    
    if texto.startswith('"') and texto.endswith('"'):
        texto = texto[1:-1]
    
    texto = texto.replace('""', '"')
    
    try:
        dados_json = json.loads(texto)
        if isinstance(dados_json, dict):
            texto = " ".join([str(v) for v in dados_json.values()])
    except Exception:
        pass

    texto = texto.lower()
    texto = urllib.parse.unquote(texto)
    return texto

def analisar_logs_neural_network():
    print("\n--- [PASSO 3: ANALISANDO LOGS REAIS COM MODELO HÍBRIDO OTIMIZADO] ---")
    
    LOG_PATH = 'data/app_logs.log.txt'
    MODEL_PATH = 'models/model_nn.keras'
    VECTORIZER_PATH = 'models/tfidf_vectorizer.joblib'
    OUTPUT_REPORT = 'reports/resultado_analise.csv'
    
    if not os.path.exists(LOG_PATH):
        print(f"Erro: Arquivo de logs em {LOG_PATH} não encontrado.")
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Erro: Modelo treinado ou Vetorizador TF-IDF não encontrados. Execute o script de treino primeiro.")
        return

    print("Carregando Rede Neural e Vetorizador TF-IDF de Caracteres...")
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print("Carregando e fatiando arquivo de logs textuais...")
    linhas_processadas = []
    padrao_log_complexo = re.compile(r"^([^\s,]+)\s+INFO\s+IP:\s+([^\s]+)\s+BODY:\s+(.*)$", re.IGNORECASE)
    
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for num_linha, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha:
                    continue
                
                if num_linha == 1 and ("data/hora" in linha.lower() or "payload" in linha.lower()):
                    continue
                
                match = padrao_log_complexo.match(linha)
                if match:
                    linhas_processadas.append([match.group(1), match.group(2), match.group(3)])
                else:
                    partes = linha.split(',', 2)
                    if len(partes) >= 3:
                        linhas_processadas.append([partes[0], partes[1], partes[2]])
                    else:
                        linhas_processadas.append(["Desconhecido", "Desconhecido", linha])
                        
        df_logs = pd.DataFrame(linhas_processadas, columns=['Data/Hora', 'IP', 'Payload'])
        print(f"Total de {len(df_logs)} entradas de log carregadas.")
        
    except Exception as e:
        print(f"Erro fatal ao processar o arquivo de logs: {e}")
        return

    print("Normalizando e extraindo features via NLP automático...")
    payloads_limpos = [limpar_e_normalizar_log(row['Payload']) for _, row in df_logs.iterrows()]
    
    X_features = vectorizer.transform(payloads_limpos).toarray()

    print("Submetendo dados à Rede Neural...")
    predicoes_prob = model.predict(X_features, verbose=0).flatten()

    print("Aplicando regras híbridas de tomada de decisão (IA Ativa + Cinto Regex)...")
    classificacoes = []
    confiancas = []
    tipos_provaveis = []

    # REGEX
    regex_xss = re.compile(
        r"<\s*script|<img[/\s]+[^>]*on\w+\s*=|<\s*\w+[/\s]+[^>]*on\w+\s*="
        r"|href\s*=\s*['\"]?\s*javascript:"
        r"|src\s*=\s*['\"]?\s*javascript:"
        r"|background\s*=\s*['\"]?\s*javascript:"
        r"|action\s*=\s*['\"]?\s*javascript:"
        r"|formaction\s*=\s*['\"]?\s*javascript:"
        r"|<\s*(object|embed|iframe|link|isindex|keygen)\b"
        r"|<\s*form[/\s]+[^>]*action\s*=\s*['\"]?\s*javascript:",
        re.IGNORECASE
    )
    regex_sqli = re.compile(r"'\s*or\s*['\"\d]|--|/\*|\b(union|insert|update|drop)\b", re.IGNORECASE)

    THRESHOLD_PADRAO = 0.15

    for i, prob in enumerate(predicoes_prob):
        payload_lower = payloads_limpos[i]
        
        detectado_por_regex_xss = bool(regex_xss.search(payload_lower))
        detectado_por_regex_sqli = bool(regex_sqli.search(payload_lower))
        
        eh_fluxo_cadastro = any(w in payload_lower for w in ['usuario', 'bibliotecario', 'gmail.com', 'email.com.br', 'telefone'])
        threshold_atual = 0.70 if eh_fluxo_cadastro else THRESHOLD_PADRAO
        
        ia_deu_ataque = (prob > threshold_atual)
        regex_detectou = (detectado_por_regex_xss or detectado_por_regex_sqli)
        
        # --- MATRIZ DE DECISÃO ---
        if regex_detectou:
            classificacoes.append("ATAQUE")
            confianca_percentual = 100.0
            
            # Se a IA também suspeitou do log, vira Redundante
            if prob > 0.01:
                if detectado_por_regex_sqli:
                    tipos_provaveis.append("SQLI (Regex_Redundante)")
                else:
                    tipos_provaveis.append("XSS (Regex_Redundante)")
            else:
                # Se a IA ignorou por completo, foi ponto cego e o Regex salvou isolado
                if detectado_por_regex_sqli:
                    tipos_provaveis.append("SQLI (Filtro_regex)")
                else:
                    tipos_provaveis.append("XSS (Filtro_regex)")
                    
        elif ia_deu_ataque and not regex_detectou:
            # Caso a IA pegue algo que o regex não conhece
            classificacoes.append("ATAQUE")
            confianca_percentual = prob * 100
            if any(w in payload_lower for w in ['select', 'union', 'where', 'or 1=1', '--', '/*']):
                tipos_provaveis.append("SQLI (Detectado_IA)")
            elif any(w in payload_lower for w in ['<script', '<svg', '<img', 'onerror', 'onload', 'javascript:', 'action']):
                tipos_provaveis.append("XSS (Detectado_IA)")
            else:
                tipos_provaveis.append("ANOMALIA (Detectado_IA)")
                
        else:
            classificacoes.append("NORMAL")
            confianca_percentual = (1 - prob) * 100
            tipos_provaveis.append("Normal")
            
        confiancas.append(f"{confianca_percentual:.1f}%")

    df_logs['Classificacao'] = classificacoes
    df_logs['Confianca'] = confiancas
    df_logs['Tipo_Provavel'] = tipos_provaveis

    if not os.path.exists('reports'): 
        os.makedirs('reports')
    df_logs.to_csv(OUTPUT_REPORT, index=False)
    print(f"\n[SUCESSO] Análise concluída! Relatório atualizado em: {OUTPUT_REPORT}")

if __name__ == "__main__":
    analisar_logs_neural_network()