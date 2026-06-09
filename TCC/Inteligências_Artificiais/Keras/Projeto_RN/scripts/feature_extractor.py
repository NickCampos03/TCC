import re
import html
import urllib.parse


FEATURE_COLUMNS = [
    "url_length", "url_special_characters", "url_tag_script", "url_tag_iframe", "url_attr_src",
    "url_event_onload", "url_event_onmouseover", "url_cookie", "url_number_keywords_param",
    "url_number_domain", "html_tag_script", "html_tag_iframe", "html_tag_meta", "html_tag_object",
    "html_tag_embed", "html_tag_link", "html_tag_svg", "html_tag_frame", "html_tag_form",
    "html_tag_div", "html_tag_style", "html_tag_img", "html_tag_input", "html_tag_textarea",
    "html_attr_action", "html_attr_background", "html_attr_classid", "html_attr_codebase",
    "html_attr_href", "html_attr_longdesc", "html_attr_profile", "html_attr_src", "html_attr_usemap",
    "html_attr_http-equiv", "html_event_onblur", "html_event_onchange", "html_event_onclick",
    "html_event_onerror", "html_event_onfocus", "html_event_onkeydown", "html_event_onkeypress",
    "html_event_onkeyup", "html_event_onload", "html_event_onmousedown", "html_event_onmouseout",
    "html_event_onmouseover", "html_event_onmouseup", "html_event_onsubmit", "html_number_keywords_evil",
    "js_file", "js_pseudo_protocol", "js_dom_location", "js_dom_document", "js_prop_cookie",
    "js_prop_referrer", "js_method_write", "js_method_getElementsByTagName",
    "js_method_getElementById", "js_method_alert", "js_method_eval", "js_method_fromCharCode",
    "js_method_confirm", "js_min_length", "js_min_define_function", "js_min_function_calls",
    "js_string_max_length", "html_length",
]


XSS_INDICATOR_PATTERNS = [
    r"<\s*/?\s*(script|iframe|svg|img|body|input|video|audio|source|details|marquee|frameset|a)\b",
    r"\bon\w+\s*=",
    r"\bjavascript\s*:",
    r"\bdata\s*:\s*text/html\s*;\s*base64\s*,",
    r"\bdocument\s*\.\s*(cookie|write|location|referrer)\b",
    r"\b(window\s*\.\s*)?location\s*=",
    r"\b(eval|alert|confirm|prompt|fetch|atob|btoa)\s*\(",
    r"\bsendbeacon\s*\(",
    r"\bfromcharcode\s*\(",
    r"\blocalstorage\b",
]


def normalize_payload(payload):
    decoded = urllib.parse.unquote(str(payload).lower())
    decoded = re.sub(
        r"\\u([0-9a-f]{4})",
        lambda match: chr(int(match.group(1), 16)),
        decoded,
        flags=re.IGNORECASE,
    )
    return html.unescape(decoded)


def xss_indicator_score(payload):
    p = normalize_payload(payload)
    score = sum(1 for pattern in XSS_INDICATOR_PATTERNS if re.search(pattern, p))

    if "<" in p and ">" in p:
        score += 1
    if "cookie" in p and ("document" in p or "fetch" in p or "sendbeacon" in p):
        score += 1
    if "http" in p and any(token in p for token in ["evil", "attacker", "malware", "phish", "malicious"]):
        score += 1

    return score


def extract_features(payload):
    p = normalize_payload(payload)
    features = {k: 0 for k in FEATURE_COLUMNS}

    features["url_length"] = len(p)
    features["html_length"] = len(p)
    features["url_special_characters"] = len(re.findall(r"[\'\"\;#\*<>\(\)\[\]\{\}\=\-\&\%\?\!]", p))

    features["url_number_domain"] = len(re.findall(r"https?://|www\.", p))
    features["url_number_keywords_param"] = len(re.findall(r"[\?\&]\w+\=", p))
    features["url_cookie"] = p.count("cookie")

    tag_patterns = [
        ("html_tag_script", r"<\s*script"), ("html_tag_iframe", r"<\s*iframe"),
        ("html_tag_meta", r"<\s*meta"), ("html_tag_object", r"<\s*object"),
        ("html_tag_embed", r"<\s*embed"), ("html_tag_link", r"<\s*link"),
        ("html_tag_svg", r"<\s*svg"), ("html_tag_frame", r"<\s*frame"),
        ("html_tag_form", r"<\s*form"), ("html_tag_div", r"<\s*div"),
        ("html_tag_style", r"<\s*style"), ("html_tag_img", r"<\s*img"),
        ("html_tag_input", r"<\s*input"), ("html_tag_textarea", r"<\s*textarea"),
    ]
    for feature_name, pattern in tag_patterns:
        features[feature_name] = len(re.findall(pattern, p))

    features["url_tag_script"] = features["html_tag_script"]
    features["url_tag_iframe"] = features["html_tag_iframe"]

    attributes = ["action", "background", "classid", "codebase", "href", "longdesc", "profile", "src", "usemap", "http-equiv"]
    for attr in attributes:
        features[f"html_attr_{attr}"] = len(re.findall(rf"\b{attr}\s*=", p))
    features["url_attr_src"] = features["html_attr_src"]

    events = [
        "onblur", "onchange", "onclick", "onerror", "onfocus", "onkeydown", "onkeypress",
        "onkeyup", "onload", "onmousedown", "onmouseout", "onmouseover", "onmouseup", "onsubmit",
    ]
    for event in events:
        features[f"html_event_{event}"] = len(re.findall(rf"\b{event}\s*=", p))
    features["url_event_onload"] = features["html_event_onload"]
    features["url_event_onmouseover"] = features["html_event_onmouseover"]

    evil_keywords = [
        "javascript:", "expression", "prompt", "string.fromcharcode", "base64",
        "document.write", "window.location", "innerhtml", "outerhtml", "payload",
    ]
    features["html_number_keywords_evil"] = sum(p.count(keyword) for keyword in evil_keywords)

    features["js_file"] = int(".js" in p or "text/javascript" in p)
    features["js_pseudo_protocol"] = p.count("javascript:")
    features["js_dom_location"] = len(re.findall(r"\b(window\.)?location\b", p))
    features["js_dom_document"] = len(re.findall(r"\bdocument\b", p))
    features["js_prop_cookie"] = len(re.findall(r"\bdocument\.cookie\b", p))
    features["js_prop_referrer"] = len(re.findall(r"\bdocument\.referrer\b", p))
    features["js_method_write"] = len(re.findall(r"\bdocument\.write\(", p))
    features["js_method_getElementsByTagName"] = len(re.findall(r"\bgetelementsbytagname\(", p))
    features["js_method_getElementById"] = len(re.findall(r"\bgetelementbyid\(", p))
    features["js_method_alert"] = len(re.findall(r"\balert\(", p))
    features["js_method_eval"] = len(re.findall(r"\beval\(", p))
    features["js_method_fromCharCode"] = len(re.findall(r"\bfromcharcode\(", p))
    features["js_method_confirm"] = len(re.findall(r"\bconfirm\(", p))

    strings = re.findall(r"'(.*?)'|\"(.*?)\"", p)
    if strings:
        features["js_string_max_length"] = max(max(len(single), len(double)) for single, double in strings)

    features["js_min_length"] = len(p)
    features["js_min_define_function"] = len(re.findall(r"\bfunction\s+\w*\(", p))
    features["js_min_function_calls"] = len(re.findall(r"\w+\(", p))

    return [features[column] for column in FEATURE_COLUMNS]
