import pandas as pd
import numpy as np
import os
import re
import joblib
import tensorflow as tf
import urllib.parse
import json
from scripts.text_features import enriquecer_payload

def extrair_texto_log(payload):
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
    original = texto
    for _ in range(3):
        decodificado = urllib.parse.unquote(texto)
        if decodificado == texto:
            break
        texto = decodificado
    return f'{texto} {original}' if texto != original else texto


def limpar_e_normalizar_log(payload):
    """Prepara a entrada da MLP com os descritores usados no treinamento."""
    texto = extrair_texto_log(payload)
    texto_enriquecido = enriquecer_payload(texto)

    # O dataset XSS existente contém descritores numéricos convertidos em texto
    # (por exemplo, "html tag script count 1"). Reproduzimos esse formato nos
    # logs para que a MLP receba sinais na mesma representação do treinamento.
    features_log = []
    tamanho_url = min(len(texto) // 10, 50)
    tamanho_html = min(len(texto) // 10, 50)
    especiais = len(re.findall(r"[\'\";#*<>()[\]{}=\-]", texto))
    features_log.extend([
        f'url length bucket {tamanho_url}',
        f'url special characters bucket {min(especiais // 10, 50)}',
        f'html length bucket {tamanho_html}',
    ])

    tags_html = (
        'script', 'iframe', 'meta', 'object', 'embed', 'link', 'svg', 'frame',
        'form', 'div', 'style', 'img', 'input', 'textarea',
    )
    for tag in tags_html:
        if re.search(rf'<\s*{tag}\b', texto, re.IGNORECASE):
            features_log.append(f'html tag {tag} count 1')
            if tag in {'script', 'iframe'}:
                features_log.append(f'url tag {tag} count 1')

    eventos_html = (
        'onblur', 'onchange', 'onclick', 'onerror', 'onfocus', 'onkeydown',
        'onkeypress', 'onkeyup', 'onload', 'onmousedown', 'onmouseout',
        'onmouseover', 'onmouseup', 'onsubmit',
    )
    for evento in eventos_html:
        if re.search(rf'\b{evento}\s*=', texto, re.IGNORECASE):
            features_log.append(f'html event {evento} count 1')

    atributos_html = (
        'action', 'background', 'classid', 'codebase', 'href', 'longdesc',
        'profile', 'src', 'usemap', 'http-equiv',
    )
    for atributo in atributos_html:
        if re.search(rf'\b{re.escape(atributo)}\s*=', texto, re.IGNORECASE):
            features_log.append(f'html attr {atributo} count 1')
            if atributo == 'src':
                features_log.append('url attr src count 1')

    if 'cookie' in texto:
        features_log.append('url cookie count 1')

    if re.search(r'javascript\s*:', texto, re.IGNORECASE):
        features_log.append('js pseudo protocol count 1')
    if 'document.cookie' in texto:
        features_log.append('js prop cookie count 1')
    if 'document.referrer' in texto:
        features_log.append('js prop referrer count 1')
    if re.search(r'(document|window)\.location|location\.href', texto, re.IGNORECASE):
        features_log.append('js dom location count 1')
    if 'document.write' in texto:
        features_log.append('js method write count 1')
    if re.search(r'\balert\s*\(', texto, re.IGNORECASE):
        features_log.append('js method alert count 1')
    if re.search(r'\beval\s*\(', texto, re.IGNORECASE):
        features_log.append('js method eval count 1')

    if features_log:
        marcadores_xss = re.findall(r'__xss_[a-z_]+__', texto_enriquecido)
        parece_sqli = bool(re.search(r"'\s*or\s*['\"\d]|--|/\*|\b(union|insert|update|drop|select)\b", texto, re.IGNORECASE))
        parece_path_traversal = bool(re.search(r"\.\.(?:/|\\)|/etc/passwd|win\.ini", texto, re.IGNORECASE))
        possui_sinais_xss = any(
            item.startswith((
                'html tag ', 'html event ', 'html attr ', 'js ', 'url tag ',
                'url event ', 'url attr ',
            ))
            for item in features_log
        )

        # Em XSS puro, usa os descritores que existem no dataset de treinamento.
        # Retém o conteúdo bruto em payloads mistos para preservar outros sinais.
        if possui_sinais_xss and not parece_sqli and not parece_path_traversal:
            return ' '.join(features_log + marcadores_xss)
        if possui_sinais_xss:
            texto_enriquecido = f"{texto_enriquecido} {' '.join(features_log)}"
    return texto_enriquecido

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
    padrao_log_spring = re.compile(
        r"^([^\s]+)\s+INFO .*?\bREQ\s+\|\s+IP:\s*(.*?)\s+\|\s*METHOD:\s*(.*?)\s+"
        r"\|\s*URL:\s*(.*?)\s+\|\s*QUERY:\s*(.*?)\s+\|\s*BODY:\s*(.*?)\s+"
        r"\|\s*STATUS:\s*(\d+)",
        re.IGNORECASE,
    )
    padrao_log_complexo = re.compile(r"^([^\s,]+)\s+INFO\s+IP:\s+([^\s]+)\s+BODY:\s+(.*)$", re.IGNORECASE)
    
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for num_linha, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha:
                    continue
                
                if num_linha == 1 and ("data/hora" in linha.lower() or "payload" in linha.lower()):
                    continue
                
                match_spring = padrao_log_spring.match(linha)
                if match_spring:
                    timestamp, ip, metodo, url, query, body, status = match_spring.groups()
                    # Exclui metadados do servidor (thread, UA, status e tempo)
                    # para que não sejam confundidos com conteúdo do ataque.
                    payload = f"URL: {url} QUERY: {query} BODY: {body}"
                    linhas_processadas.append([timestamp, ip, payload])
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
    payloads_regex = [extrair_texto_log(row['Payload']) for _, row in df_logs.iterrows()]
    payloads_limpos = [limpar_e_normalizar_log(row['Payload']) for _, row in df_logs.iterrows()]
    
    X_features = vectorizer.transform(payloads_limpos)

    print("Submetendo dados à Rede Neural...")
    predicoes_modelo = model.predict(X_features, verbose=0)
    saida_multiclasse = len(predicoes_modelo.shape) == 2 and predicoes_modelo.shape[1] > 1

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
    regex_path_traversal = re.compile(
        r"(?:\.{2,}(?:[/\\]|\.{2,}[/\\])|%2e%2e(?:%2f|/|\\)|%c0%ae|%c0%af|%c1%9c"
        r"|/etc/(?:passwd|hosts|shadow)\b|/(?:config/)?application\.properties\b|/(?:arquivos|senhas)\b"
        r"|(?:^|[/\\])win\.ini\b|windows[/\\]system32)",
        re.IGNORECASE,
    )

    THRESHOLD_PADRAO = 0.15

    for i in range(len(payloads_limpos)):
        # Regras de regex continuam avaliando o payload bruto normalizado;
        # a MLP recebe os descritores compatíveis com o treinamento.
        payload_lower = payloads_regex[i]

        if saida_multiclasse:
            probs = predicoes_modelo[i]
            classe_ia = int(np.argmax(probs))
            confianca_ia = float(np.max(probs))
            prob_ataque = float(1 - probs[0])
        else:
            prob_ataque = float(predicoes_modelo[i][0] if len(predicoes_modelo.shape) == 2 else predicoes_modelo[i])
            classe_ia = 1 if prob_ataque > 0.5 else 0
            confianca_ia = prob_ataque if classe_ia != 0 else 1 - prob_ataque

        
        detectado_por_regex_xss = bool(regex_xss.search(payload_lower))
        detectado_por_regex_sqli = bool(regex_sqli.search(payload_lower))
        detectado_por_regex_path = bool(regex_path_traversal.search(payload_lower))
        
        eh_fluxo_cadastro = any(w in payload_lower for w in ['usuario', 'bibliotecario', 'gmail.com', 'email.com.br', 'telefone'])
        threshold_atual = 0.70 if eh_fluxo_cadastro else THRESHOLD_PADRAO
        
        ia_deu_ataque = (classe_ia != 0 and confianca_ia > threshold_atual) if saida_multiclasse else (prob_ataque > threshold_atual)
        regex_detectou = (detectado_por_regex_xss or detectado_por_regex_sqli or detectado_por_regex_path)
        classe_regex = (
            3 if detectado_por_regex_path
            else (2 if detectado_por_regex_sqli else (1 if detectado_por_regex_xss else 0))
        )
        
        # --- MATRIZ DE DECISÃO ---
        if regex_detectou:
            classificacoes.append("ATAQUE")
            confianca_percentual = 100.0
            
            # Redundante significa que a IA previu a mesma classe detectada pelo regex.
            if classe_ia == classe_regex and confianca_ia > threshold_atual:
                if detectado_por_regex_path:
                    tipos_provaveis.append("PATH_TRAVERSAL (Regex_Redundante)")
                elif detectado_por_regex_sqli:
                    tipos_provaveis.append("SQLI (Regex_Redundante)")
                else:
                    tipos_provaveis.append("XSS (Regex_Redundante)")
            else:
                # Se a IA ignorou por completo, foi ponto cego e o Regex salvou isolado
                if detectado_por_regex_path:
                    tipos_provaveis.append("PATH_TRAVERSAL (Filtro_regex)")
                elif detectado_por_regex_sqli:
                    tipos_provaveis.append("SQLI (Filtro_regex)")
                else:
                    tipos_provaveis.append("XSS (Filtro_regex)")
                    
        elif ia_deu_ataque and not regex_detectou:
            # Caso a IA pegue algo que o regex não conhece
            classificacoes.append("ATAQUE")
            confianca_percentual = confianca_ia * 100
            if saida_multiclasse and classe_ia == 3:
                tipos_provaveis.append("PATH_TRAVERSAL (Detectado_IA)")
            elif saida_multiclasse and classe_ia == 2:
                tipos_provaveis.append("SQLI (Detectado_IA)")
            elif saida_multiclasse and classe_ia == 1:
                tipos_provaveis.append("XSS (Detectado_IA)")
            elif any(w in payload_lower for w in ['select', 'union', 'where', 'or 1=1', '--', '/*']):
                tipos_provaveis.append("SQLI (Detectado_IA)")
            elif any(w in payload_lower for w in ['<script', '<svg', '<img', 'onerror', 'onload', 'javascript:', 'action']):
                tipos_provaveis.append("XSS (Detectado_IA)")
            else:
                tipos_provaveis.append("ANOMALIA (Detectado_IA)")
                
        else:
            classificacoes.append("NORMAL")
            confianca_percentual = (1 - prob_ataque) * 100
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
