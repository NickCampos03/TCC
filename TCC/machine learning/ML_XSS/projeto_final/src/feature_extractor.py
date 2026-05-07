import re

def extrair_features(payload):
    """
    Extração de características baseada na lógica do Dataset 1 (XSS Engineered).
    Retorna uma lista de 20 valores numéricos.
    """
    if not isinstance(payload, str):
        payload = str(payload)
    
    payload = payload.lower()
    
    features = {
        # --- ESTRUTURA E TAMANHO ---
        'comprimento': len(payload),
        'caracteres_especiais': len(re.findall(r'[<>\?&%=()\'\"\\/]', payload)),
        'encoding_hex': len(re.findall(r'%[0-9a-f]{2}', payload)),
        
        # --- TAGS HTML CRÍTICAS ---
        'tag_script': len(re.findall(r'<script', payload)),
        'tag_iframe': len(re.findall(r'<iframe', payload)),
        'tag_svg': len(re.findall(r'<svg', payload)),
        'tag_img': len(re.findall(r'<img', payload)),
        'tag_body': len(re.findall(r'<body', payload)),
        'tag_form': len(re.findall(r'<form', payload)),
        
        # --- ATRIBUTOS PERIGOSOS ---
        'attr_src': len(re.findall(r'src\s*=', payload)),
        'attr_href': len(re.findall(r'href\s*=', payload)),
        'attr_style': len(re.findall(r'style\s*=', payload)),
        
        # --- EVENTOS JAVASCRIPT (DOM) ---
        'eventos_js': len(re.findall(r'on\w+\s*=', payload)), # ex: onload, onerror
        
        # --- FUNÇÕES E MÉTODOS JS ---
        'js_alert_confirm': len(re.findall(r'(alert|confirm|prompt)\s*\(', payload)),
        'js_eval': len(re.findall(r'eval\s*\(', payload)),
        'js_document_write': len(re.findall(r'document\.write', payload)),
        'js_string_fromchar': len(re.findall(r'fromcharcode', payload)),
        
        # --- ACESSO A DADOS ---
        'js_cookie': len(re.findall(r'document\.cookie', payload)),
        'js_location': len(re.findall(r'location\.', payload)),
        'js_exfiltracao': len(re.findall(r'(fetch|xmlhttprequest|sendbeacon|atob)', payload))
    }
    
    return list(features.values())