import re
import html
import urllib.parse

FEATURE_COLUMNS = [
    "url_length", "url_special_characters", "url_tag_script", "url_tag_iframe", "url_attr_src", "url_event_onload", "url_event_onmouseover", "url_cookie", "url_number_keywords_param", "url_number_domain",
    "html_tag_script", "html_tag_iframe", "html_tag_meta", "html_tag_object", "html_tag_embed", "html_tag_link", "html_tag_svg", "html_tag_frame", "html_tag_form", "html_tag_div", "html_tag_style", "html_tag_img", "html_tag_input", "html_tag_textarea",
    "html_attr_action", "html_attr_background", "html_attr_classid", "html_attr_codebase", "html_attr_href", "html_attr_longdesc", "html_attr_profile", "html_attr_src", "html_attr_usemap", "html_attr_http-equiv",
    "html_event_onblur", "html_event_onchange", "html_event_onclick", "html_event_onerror", "html_event_onfocus", "html_event_onkeydown", "html_event_onkeypress", "html_event_onkeyup", "html_event_onload", "html_event_onmousedown", "html_event_onmouseout", "html_event_onmouseover", "html_event_onmouseup", "html_event_onsubmit",
    "html_number_keywords_evil",
    "js_file", "js_pseudo_protocol", "js_dom_location", "js_dom_document", "js_prop_cookie", "js_prop_referrer", "js_method_write", "js_method_getElementsByTagName", "js_method_getElementById", "js_method_alert", "js_method_eval", "js_method_fromCharCode", "js_method_confirm",
    "js_min_length", "js_min_define_function", "js_min_function_calls", "js_string_max_length",
    "html_length",
    "sqli_keyword_select", "sqli_keyword_union", "sqli_keyword_insert", "sqli_keyword_update", "sqli_keyword_delete", "sqli_keyword_drop", "sqli_keyword_create", "sqli_keyword_alter", "sqli_keyword_exec",
    "sqli_comment_inline", "sqli_comment_block", "sqli_quote_single", "sqli_quote_double", "sqli_semicolon", "sqli_boolean_or", "sqli_boolean_and", "sqli_tautology", "sqli_sleep", "sqli_benchmark",
    "sqli_information_schema", "sqli_system_tables", "sqli_concat", "sqli_hex", "sqli_version", "sqli_error_based", "sqli_stacked_queries", "sqli_parentheses", "sqli_operator_equals",
    # PATH TRAVERSAL FEATURES — adicionadas para a quarta classe do modelo.
    "path_traversal_dotdot", "path_traversal_encoded_dotdot", "path_traversal_double_encoded",
    "path_traversal_separator", "path_traversal_system_unix", "path_traversal_system_windows",
    "path_traversal_sensitive_file"
]

EVIL_KEYWORDS = [
    "alert", "eval", "prompt", "confirm", "script", "iframe", "document.cookie", "fromcharcode", "onerror", "onload", "onclick", "onmouseover", "ontoggle", "onpageshow", "onstart", "javascript:", "vbscript:",
    "data:text/html", "xss", "evil.com", "attacker", "malware", "phish", "malicious", "fetch", "xmlhttprequest", "sendbeacon", "localstorage", "atob", "btoa", "cookie", "location", "write"
]

def count(pattern, text):
    return len(re.findall(pattern, text, re.IGNORECASE))


def extract_features(payload):
    if payload is None:
        payload = ""

    payload = str(payload)
    decoded = html.unescape(urllib.parse.unquote(payload))
    lower = decoded.lower()
    features = {}

    # 1. URL FEATURES
    features["url_length"] = float(len(payload))
    features["url_special_characters"] = float(len(re.findall(r'[<>\?&%=()\'\"\\/]', lower)))
    features["url_tag_script"] = float(count(r"<script", lower))
    features["url_tag_iframe"] = float(count(r"<iframe", lower))
    features["url_attr_src"] = float(count(r"src\s*=", lower))
    features["url_event_onload"] = float(count(r"onload\s*=", lower))
    features["url_event_onmouseover"] = float(count(r"onmouseover\s*=", lower))
    features["url_cookie"] = float(count(r"cookie", lower))
    features["url_number_keywords_param"] = float(count(r"(id|query|search|name|text|msg|payload)\s*=", lower))
    features["url_number_domain"] = float(count(r"https?://", lower))

    # 2. HTML TAGS
    tags = ["script", "iframe", "meta", "object", "embed", "link", "svg", "form", "div", "style", "img", "input", "textarea"]
    for tag in tags:
        features[f"html_tag_{tag}"] = float(count(rf"<{tag}", lower))
    features["html_tag_frame"] = float(count(r"<frame|<frameset", lower))

    # 3. HTML ATTRIBUTES
    attrs = ["action", "background", "classid", "codebase", "href", "longdesc", "profile", "src", "usemap", "http-equiv"]
    for attr in attrs:
        features[f"html_attr_{attr}"] = float(count(rf"{re.escape(attr)}\s*=", lower))

    # 4. HTML EVENTS
    events = ["blur", "change", "click", "error", "focus", "keydown", "keypress", "keyup", "load", "mousedown", "mouseout", "mouseover", "mouseup", "submit"]
    for event in events:
        features[f"html_event_on{event}"] = float(count(rf"on{event}\s*=", lower))

    # 5. MALICIOUS KEYWORDS
    features["html_number_keywords_evil"] = float(sum(lower.count(keyword) for keyword in EVIL_KEYWORDS))

    # 6. JAVASCRIPT FEATURES
    features["js_file"] = float(count(r"\.js\b", lower))
    features["js_pseudo_protocol"] = float(count(r"javascript:|vbscript:", lower))
    features["js_dom_location"] = float(count(r"document\.location|window\.location|\blocation\s*[.=]", lower))
    features["js_dom_document"] = float(count(r"document\.", lower))
    features["js_prop_cookie"] = float(count(r"document\.cookie", lower))
    features["js_prop_referrer"] = float(count(r"document\.referrer", lower))
    features["js_method_write"] = float(count(r"document\.write\s*\(", lower))
    features["js_method_getElementsByTagName"] = float(count(r"getelementsbytagname", lower))
    features["js_method_getElementById"] = float(count(r"getelementbyid", lower))
    features["js_method_alert"] = float(count(r"\balert\s*\(", lower))
    features["js_method_eval"] = float(count(r"\beval\s*\(", lower))
    features["js_method_fromCharCode"] = float(count(r"fromcharcode", lower))
    features["js_method_confirm"] = float(count(r"\bconfirm\s*\(", lower))

    # 7. JS STATISTICS
    features["js_min_length"] = float(len(lower))
    features["js_min_define_function"] = float(count(r"\bfunction\b", lower))
    features["js_min_function_calls"] = float(len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(", lower)))

    strings = re.findall(r"'([^']*)'|\"([^\"]*)\"", decoded)
    max_len = 0
    for pair in strings:
        for value in pair:
            if value:
                max_len = max(max_len, len(value))
                
    features["js_string_max_length"] = float(max_len)
    features["html_length"] = float(len(decoded))

    # 8. SQL INJECTION FEATURES
    features["sqli_keyword_select"] = float(count(r"\bselect\b", lower))
    features["sqli_keyword_union"] = float(count(r"\bunion\b", lower))
    features["sqli_keyword_insert"] = float(count(r"\binsert\b", lower))
    features["sqli_keyword_update"] = float(count(r"\bupdate\b", lower))
    features["sqli_keyword_delete"] = float(count(r"\bdelete\b", lower))
    features["sqli_keyword_drop"] = float(count(r"\bdrop\b", lower))
    features["sqli_keyword_create"] = float(count(r"\bcreate\b", lower))
    features["sqli_keyword_alter"] = float(count(r"\balter\b", lower))
    features["sqli_keyword_exec"] = float(count(r"\bexec(?:ute)?\b|xp_cmdshell", lower))
    features["sqli_comment_inline"] = float(count(r"(--|#|%23)", lower))
    features["sqli_comment_block"] = float(count(r"/\*|\*/", lower))
    features["sqli_quote_single"] = float(lower.count("'"))
    features["sqli_quote_double"] = float(lower.count('"'))
    features["sqli_semicolon"] = float(lower.count(";"))
    features["sqli_boolean_or"] = float(count(r"\bor\b\s+[\w'\"()]+?\s*=", lower))
    features["sqli_boolean_and"] = float(count(r"\band\b\s+[\w'\"()]+?\s*=", lower))
    features["sqli_tautology"] = float(count(r"(['\"]?\s*\d+\s*['\"]?\s*=\s*['\"]?\s*\d+)|(['\"]\s*=\s*['\"])", lower))
    features["sqli_sleep"] = float(count(r"\b(pg_sleep|sleep)\s*\(|waitfor\s+delay", lower))
    features["sqli_benchmark"] = float(count(r"\bbenchmark\s*\(", lower))
    features["sqli_information_schema"] = float(count(r"information_schema", lower))
    features["sqli_system_tables"] = float(count(r"sys\.all_tables|sysobjects|sqlite_master|pg_catalog", lower))
    features["sqli_concat"] = float(count(r"\bconcat\s*\(|\|\|", lower))
    features["sqli_hex"] = float(count(r"0x[0-9a-f]+", lower))
    features["sqli_version"] = float(count(r"@@version|\bversion\s*\(", lower))
    features["sqli_error_based"] = float(count(r"extractvalue|updatexml|utl_inaddr|dbms_pipe|ora-|sql syntax|mysql", lower))
    features["sqli_stacked_queries"] = float(count(r";\s*(select|insert|update|delete|drop|create|alter|exec)\b", lower))
    features["sqli_parentheses"] = float(lower.count("(") + lower.count(")"))
    features["sqli_operator_equals"] = float(lower.count("="))

    # 9. PATH TRAVERSAL FEATURES — nova vetorizacao, sem alterar as features XSS/SQLi.
    raw_lower = payload.lower()
    features["path_traversal_dotdot"] = float(len(re.findall(r"\.\.[\\/]", decoded)))
    features["path_traversal_encoded_dotdot"] = float(len(re.findall(r"%(?:2e|252e)%(?:2e|252e)%(?:2f|5c|252f|255c)", raw_lower)))
    # Somente escapes duplos de separadores ou pontos sao sinais de traversal.
    # Contar todo "%2f" ou toda barra confundia tags XSS e URLs benignas com Path Traversal.
    features["path_traversal_double_encoded"] = float(len(re.findall(r"%25(?:2e|2f|5c)", raw_lower)))
    # Separadores so contam quando fazem parte de um salto de diretorio (../ ou ..\).
    features["path_traversal_separator"] = float(len(re.findall(r"\.\.[\\/]", decoded)))
    features["path_traversal_system_unix"] = float(count(r"/(?:etc/(?:passwd|shadow|hosts)|proc/|var/log/|usr/|home/)", lower))
    features["path_traversal_system_windows"] = float(count(r"(?:[a-z]:)?[\\/](?:windows[\\/]|winnt[\\/]|boot\.ini)", lower))
    features["path_traversal_sensitive_file"] = float(count(r"(?:passwd|shadow|boot\.ini|win\.ini|system\.ini|hosts)", lower))
    return [features[col] for col in FEATURE_COLUMNS]


def xss_indicator_score(features):
    idx_script = FEATURE_COLUMNS.index("html_tag_script")
    idx_iframe = FEATURE_COLUMNS.index("html_tag_iframe")
    idx_onerror = FEATURE_COLUMNS.index("html_event_onerror")
    idx_onload = FEATURE_COLUMNS.index("html_event_onload")
    idx_eval = FEATURE_COLUMNS.index("js_method_eval")
    idx_alert = FEATURE_COLUMNS.index("js_method_alert")
    idx_fromcharcode = FEATURE_COLUMNS.index("js_method_fromCharCode")
    idx_cookie = FEATURE_COLUMNS.index("js_prop_cookie")
    idx_link = FEATURE_COLUMNS.index("html_tag_link")
    idx_style = FEATURE_COLUMNS.index("html_tag_style")
    idx_href = FEATURE_COLUMNS.index("html_attr_href")
    idx_domain = FEATURE_COLUMNS.index("url_number_domain")
    idx_evil = FEATURE_COLUMNS.index("html_number_keywords_evil")
    idx_pseudo = FEATURE_COLUMNS.index("js_pseudo_protocol")
    idx_location = FEATURE_COLUMNS.index("js_dom_location")

    score = (
        features[idx_script] * 3 + features[idx_iframe] * 3 + features[idx_onerror] * 2 + features[idx_onload] * 2 +
        features[idx_eval] * 4 + features[idx_alert] * 2 + features[idx_fromcharcode] * 2 + features[idx_cookie] * 3 +
        features[idx_pseudo] * 3 + features[idx_location] * 2 + features[idx_link] * 2 + features[idx_style] * 2 +
        features[idx_href] + features[idx_domain] + min(features[idx_evil], 8)
    )

    return float(score)


def sqli_indicator_score(features):
    feature_map = dict(zip(FEATURE_COLUMNS, features))
    keyword_score = sum(feature_map.get(name, 0.0) for name in [
        "sqli_keyword_select", "sqli_keyword_union", "sqli_keyword_drop", "sqli_keyword_exec",
        "sqli_information_schema", "sqli_system_tables", "sqli_version", "sqli_error_based"
    ])

    score = (
        feature_map.get("sqli_keyword_union", 0.0) * 4 +
        feature_map.get("sqli_tautology", 0.0) * 4 +
        feature_map.get("sqli_sleep", 0.0) * 5 +
        feature_map.get("sqli_benchmark", 0.0) * 4 +
        feature_map.get("sqli_comment_inline", 0.0) * 3 +
        feature_map.get("sqli_comment_block", 0.0) * 2 +
        feature_map.get("sqli_boolean_or", 0.0) * 3 +
        feature_map.get("sqli_boolean_and", 0.0) * 2 +
        feature_map.get("sqli_stacked_queries", 0.0) * 4 +
        feature_map.get("sqli_error_based", 0.0) * 4 +
        feature_map.get("sqli_information_schema", 0.0) * 3 +
        feature_map.get("sqli_system_tables", 0.0) * 3 +
        min(keyword_score, 8)
    )

    return float(score)


def path_traversal_indicator_score(features):
    """Retorna um score explicavel para sinais de Path Traversal em logs."""
    feature_map = dict(zip(FEATURE_COLUMNS, features))
    return float(
        feature_map.get("path_traversal_dotdot", 0.0) * 3
        + feature_map.get("path_traversal_encoded_dotdot", 0.0) * 4
        + feature_map.get("path_traversal_double_encoded", 0.0) * 2
        + feature_map.get("path_traversal_system_unix", 0.0) * 5
        + feature_map.get("path_traversal_system_windows", 0.0) * 5
        + feature_map.get("path_traversal_sensitive_file", 0.0) * 3
    )
