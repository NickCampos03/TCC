import re

def extrair_features(payload):
    if not isinstance(payload, str):
        payload = str(payload)
    
    payload = payload.lower()
    
    # Extração de padrões puros (nomes simplificados)
    features = {
        'comprimento': len(payload),
        'caracteres_especiais': len(re.findall(r'[<>\?&%=()\'\"\\/]', payload)),
        'encoding_hex': len(re.findall(r'%[0-9a-f]{2}', payload)),
        'script': len(re.findall(r'<script', payload)),
        'iframe': len(re.findall(r'<iframe', payload)),
        'svg': len(re.findall(r'<svg', payload)),
        'img': len(re.findall(r'<img', payload)),
        'body': len(re.findall(r'<body', payload)),
        'form': len(re.findall(r'<form', payload)),
        'attr_src': len(re.findall(r'src\s*=', payload)),
        'attr_href': len(re.findall(r'href\s*=', payload)),
        'onerror': len(re.findall(r'onerror\s*=', payload)),
        'onload': len(re.findall(r'onload\s*=', payload)),
        'onfocus': len(re.findall(r'onfocus\s*=', payload)),
        'autofocus': len(re.findall(r'autofocus', payload)),
        'javascript_proto': len(re.findall(r'javascript:', payload)),
        'alert_confirm': len(re.findall(r'(alert|confirm|prompt|eval)\s*[\(\`\s]', payload)),
        'cookie_storage': len(re.findall(r'(cookie|localstorage|token|fetch|beacon)', payload))
    }
    return features