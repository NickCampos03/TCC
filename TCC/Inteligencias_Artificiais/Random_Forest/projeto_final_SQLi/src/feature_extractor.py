import re

def extrair_features(payload):
    if not isinstance(payload, str): 
        payload = str(payload)
    payload = payload.lower()
    
    # Retorna um dicionário com as métricas numéricas (Aumentado para cobrir o que conversamos)
    return {
        'comprimento': len(payload),
        'c_especiais': len(re.findall(r'[<>\?&%=()\'\"\\/;\-\*\#\+]', payload)),
        'encoding_hex': len(re.findall(r'%[0-9a-f]{2}', payload)),
        'script': len(re.findall(r'<script', payload)),
        'iframe': len(re.findall(r'<iframe', payload)),
        'svg': len(re.findall(r'<svg', payload)),
        'img': len(re.findall(r'<img', payload)),
        'body': len(re.findall(r'<body', payload)),
        'form': len(re.findall(r'<form', payload)),
        'attr_src': len(re.findall(r'src\s*=', payload)),
        'onerror': len(re.findall(r'onerror\s*=', payload)),
        'onload': len(re.findall(r'onload\s*=', payload)),
        'alert_confirm': len(re.findall(r'(alert|confirm|prompt|eval)\s*[\(\`\s]', payload)),
        'javascript_proto': len(re.findall(r'javascript:', payload)),
        'sqli_key': len(re.findall(r'(select|union|insert|update|delete|drop|where|from|limit)', payload)),
        'sqli_com': len(re.findall(r'(--|#|/\*)', payload)),
        'sqli_operadores': len(re.findall(r'(\s+or\s+|\s+and\s+|union\s+all)', payload)),
        'sqli_sleep': len(re.findall(r'(sleep|pg_sleep|waitfor\s+delay)', payload))
    }