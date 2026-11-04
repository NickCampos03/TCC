import re

def extract_features(payload):
    p = str(payload).lower()
    
    # Lista exata das 67 colunas do dataset XSS
    keys = [
        "url_length","url_special_characters","url_tag_script","url_tag_iframe","url_attr_src",
        "url_event_onload","url_event_onmouseover","url_cookie","url_number_keywords_param",
        "url_number_domain","html_tag_script","html_tag_iframe","html_tag_meta","html_tag_object",
        "html_tag_embed","html_tag_link","html_tag_svg","html_tag_frame","html_tag_form",
        "html_tag_div","html_tag_style","html_tag_img","html_tag_input","html_tag_textarea",
        "html_attr_action","html_attr_background","html_attr_classid","html_attr_codebase",
        "html_attr_href","html_attr_longdesc","html_attr_profile","html_attr_src","html_attr_usemap",
        "html_attr_http-equiv","html_event_onblur","html_event_onchange","html_event_onclick",
        "html_event_onerror","html_event_onfocus","html_event_onkeydown","html_event_onkeypress",
        "html_event_onkeyup","html_event_onload","html_event_onmousedown","html_event_onmouseout",
        "html_event_onmouseover","html_event_onmouseup","html_event_onsubmit","html_number_keywords_evil",
        "js_file","js_pseudo_protocol","js_dom_location","js_dom_document","js_prop_cookie",
        "js_prop_referrer","js_method_write","js_method_getElementsByTagName",
        "js_method_getElementById","js_method_alert","js_method_eval","js_method_fromCharCode",
        "js_method_confirm","js_min_length","js_min_define_function","js_min_function_calls",
        "js_string_max_length","html_length"
    ]
    
    features = {k: 0 for k in keys}
    
    # --- Preenchimento Geral ---
    features["url_length"] = len(p)
    features["url_special_characters"] = len(re.findall(r"[\'\"\;#\*<>\(\)\[\]\{\}\=\-]", p))
    
    # --- Detecção de SQLi ---
    sql_keywords = ['select', 'union', 'update', 'delete', 'insert', 'drop', 'from', 'where', 'or 1=1', 'sleep(']
    features["html_number_keywords_evil"] = sum(1 for word in sql_keywords if word in p)
    
    # --- Detecção de XSS ---
    if '<script' in p: features["html_tag_script"] += 1
    if '<iframe' in p: features["html_tag_iframe"] += 1
    if 'alert(' in p: features["js_method_alert"] += 1
    if 'onerror' in p: features["html_event_onerror"] += 1
    if 'onload' in p: features["html_event_onload"] += 1
    if 'document.cookie' in p: features["js_prop_cookie"] += 1
    
    # Retorna apenas os valores na ordem correta das 67 colunas
    return [features[k] for k in keys]