import re
import urllib.parse


SQLI_PATTERNS = {
    'sql_comment': re.compile(r"(--|#|/\*)", re.IGNORECASE),
    'sql_union': re.compile(r"\bunion\s+(all\s+)?select\b", re.IGNORECASE),
    'sql_boolean': re.compile(r"(\bor\b|\band\b)\s+['\"]?\w+['\"]?\s*(=|like)\s*['\"]?[%\w]+", re.IGNORECASE),
    'sql_tautology': re.compile(r"(\bor\b|\band\b)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+", re.IGNORECASE),
    'sql_time': re.compile(r"\b(waitfor\s+delay|sleep\s*\(|pg_sleep\s*\(|benchmark\s*\()", re.IGNORECASE),
    'sql_stack': re.compile(r";\s*(select|insert|update|delete|drop|alter|create)\b", re.IGNORECASE),
    'sql_schema': re.compile(r"\b(information_schema|sys\.|mysql\.|sqlite_master)\b", re.IGNORECASE),
    'sql_danger': re.compile(r"\b(drop|alter|truncate|create|exec|execute)\b", re.IGNORECASE),
    'sql_quote_operator': re.compile(r"['\"`]\s*(or|and|union|;|--|#|/\*)", re.IGNORECASE),
}

XSS_PATTERNS = {
    'xss_script': re.compile(r"<\s*script\b", re.IGNORECASE),
    'xss_event': re.compile(r"\bon\w+\s*=", re.IGNORECASE),
    'xss_js_uri': re.compile(r"javascript\s*:", re.IGNORECASE),
    'xss_dom': re.compile(r"\b(document\.cookie|document\.write|window\.location|eval\s*\()", re.IGNORECASE),
    'xss_tag': re.compile(r"<\s*(img|svg|iframe|object|embed|form|input)\b", re.IGNORECASE),
}

# Sinais explícitos para Path Traversal, incluindo formas codificadas comuns.
PATH_TRAVERSAL_PATTERNS = {
    'path_dotdot_slash': re.compile(r"\.{2,}(?:[/\\]|\.{2,}[/\\])", re.IGNORECASE),
    'path_encoded_dotdot_slash': re.compile(r"(?:\.\.|%2e%2e)(?:/|\\|%2f|%5c)", re.IGNORECASE),
    'path_encoded_segments': re.compile(r"%2e%2e%2f|%2e%2e/|\.\.%2f|%252e%252e%252f|%c0%ae|%c1%9c|%c0%af", re.IGNORECASE),
    'path_etc_passwd': re.compile(r"/etc/passwd", re.IGNORECASE),
    'path_etc_hosts': re.compile(r"/etc/hosts", re.IGNORECASE),
    'path_etc_shadow': re.compile(r"/etc/shadow", re.IGNORECASE),
    'path_config_application': re.compile(r"/(?:config/)?application\.properties\b", re.IGNORECASE),
    'path_sensitive_route_probe': re.compile(r"/(?:arquivos|senhas)\b", re.IGNORECASE),
    'path_win_ini': re.compile(r"(?:^|[/\\])win\.ini\b", re.IGNORECASE),
    'path_windows_system': re.compile(r"(?:windows[/\\](?:system32|win\.ini)|boot\.ini)", re.IGNORECASE),
}


def limpar_e_normalizar_texto(payload):
    texto = str(payload).lower()
    original = texto
    for _ in range(3):
        decodificado = urllib.parse.unquote(texto)
        if decodificado == texto:
            break
        texto = decodificado
    return f'{texto} {original}' if texto != original else texto


def sinais_sqli(payload):
    texto = str(payload)
    return [nome for nome, pattern in SQLI_PATTERNS.items() if pattern.search(texto)]


def sinais_xss(payload):
    texto = str(payload)
    return [nome for nome, pattern in XSS_PATTERNS.items() if pattern.search(texto)]


def sinais_path_traversal(payload):
    texto = str(payload)
    return [nome for nome, pattern in PATH_TRAVERSAL_PATTERNS.items() if pattern.search(texto)]


def parece_sqli(payload):
    return bool(sinais_sqli(payload))


def enriquecer_payload(payload):
    texto = limpar_e_normalizar_texto(payload)
    sinais_path = sinais_path_traversal(texto)
    marcadores = sinais_sqli(texto) + sinais_xss(texto) + sinais_path
    if sinais_path:
        marcadores.append('path_traversal_attack')
    if not marcadores:
        return texto
    return f"{texto} {' '.join('__' + marcador + '__' for marcador in marcadores)}"
