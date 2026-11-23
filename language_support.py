import re

import langid
import pycountry

LANGUAGE_CODE_PATTERN = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}\Z")
SCRIPT_NAMES = {
    "Arab": "Arabic",
    "Armn": "Armenian",
    "Beng": "Bengali",
    "Cyrl": "Cyrillic",
    "Deva": "Devanagari",
    "Ethi": "Ethiopic",
    "Geor": "Georgian",
    "Grek": "Greek",
    "Guru": "Gurmukhi",
    "Hans": "Simplified Chinese",
    "Hant": "Traditional Chinese",
    "Hebr": "Hebrew",
    "Jpan": "Japanese",
    "Khmr": "Khmer",
    "Knda": "Kannada",
    "Laoo": "Lao",
    "Latn": "Latin",
    "Mlym": "Malayalam",
    "Mymr": "Myanmar",
    "Orya": "Odia",
    "Sinh": "Sinhala",
    "Taml": "Tamil",
    "Telu": "Telugu",
    "Thai": "Thai",
    "Tibt": "Tibetan",
}
LANGUAGE_NAME_OVERRIDES = {
    "arb": "Arabic",
    "azb": "South Azerbaijani",
    "azj": "North Azerbaijani",
    "cmn": "Mandarin Chinese",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokmal",
    "pes": "Iranian Persian",
    "prs": "Dari",
    "uzn": "Northern Uzbek",
}
LANGUAGE_CODE_ALIASES = {
    "ar": ("arb",),
    "az": ("azj", "azb"),
    "fa": ("pes", "prs"),
    "fil": ("tgl",),
    "iw": ("heb",),
    "jw": ("jav",),
    "ku": ("kmr", "ckb"),
    "mo": ("ron",),
    "no": ("nob", "nno"),
    "tl": ("tgl",),
    "uz": ("uzn", "uzb"),
    "zh": ("zho", "cmn"),
}
PREFERRED_LANGUAGE_VARIANTS = {
    "ar": "arb_Arab",
    "az": "azj_Latn",
    "fa": "pes_Arab",
    "no": "nob_Latn",
    "pa": "pan_Guru",
    "sr": "srp_Cyrl",
    "uz": "uzn_Latn",
    "zh": "zho_Hans",
}


def get_supported_language_codes(tokenizer):
    return tuple(sorted(
        token for token in tokenizer.get_vocab()
        if LANGUAGE_CODE_PATTERN.fullmatch(token)
    ))


def get_language_labels(language_codes):
    labels = {}
    for code in language_codes:
        language_code, script_code = code.split("_", maxsplit=1)
        language = LANGUAGE_NAME_OVERRIDES.get(language_code)
        if language is None:
            entry = pycountry.languages.get(alpha_3=language_code)
            language = entry.name if entry is not None else language_code
        script = SCRIPT_NAMES.get(script_code, script_code)
        labels[code] = f"{language} ({script}, {code})"
    return labels


def detect_nllb_language(text, supported_codes):
    if len(text.strip()) < 4:
        return None

    detected_code, _ = langid.classify(text)
    language = pycountry.languages.get(alpha_2=detected_code)
    prefixes = set(LANGUAGE_CODE_ALIASES.get(detected_code, ()))
    if language is not None:
        prefixes.add(language.alpha_3)
    if detected_code in ("zh", "no", "fil", "tl", "iw", "jw", "mo"):
        prefixes.update(LANGUAGE_CODE_ALIASES[detected_code])

    candidates = [
        code for code in supported_codes
        if code.split("_", maxsplit=1)[0] in prefixes
    ]
    preferred = PREFERRED_LANGUAGE_VARIANTS.get(detected_code)
    if preferred in candidates:
        return preferred
    return candidates[0] if candidates else None
