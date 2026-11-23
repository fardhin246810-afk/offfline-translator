import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from language_support import detect_nllb_language, get_language_labels, get_supported_language_codes
from mt_config import BASE_MODEL_DIR, FINETUNED_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Offline Translator",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title("Offline Translator")
st.caption("Translate between 200+ languages · Detection and translation run offline on the hosting computer.")


def model_files_exist(model_dir):
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    return model_dir.is_dir() and (model_dir / "config.json").is_file() and has_weights


@st.cache_resource(show_spinner=False)
def load_tokenizer(model_dir):
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


@st.cache_data
def get_model_language_codes(model_dir):
    return get_supported_language_codes(load_tokenizer(model_dir))


@st.cache_resource(show_spinner=False)
def load_model(model_dir, device):
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model.to(device).eval()


def translate(text, model, tokenizer, source_language, target_language, num_beams):
    tokenizer.src_lang = source_language
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    target_id = tokenizer.convert_tokens_to_ids(target_language)
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise ValueError(f"{target_language!r} is not a language code supported by this tokenizer.")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_new_tokens=256,
            num_beams=num_beams,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


with st.sidebar:
    st.header("Settings")
    beams = st.slider("Beam search width", min_value=1, max_value=6, value=4)
    st.caption(f"Device: {DEVICE.upper()}")
    st.caption("Model files stay on the hosting computer.")

base_model_ready = model_files_exist(BASE_MODEL_DIR)
finetuned_model_ready = model_files_exist(FINETUNED_DIR)
if not base_model_ready and not finetuned_model_ready:
    st.error("The offline base model is not installed yet.")
    st.markdown("Connect to the internet once, install dependencies, and download the model:")
    st.code("pip install -r requirements.txt\npy download_model.py", language="powershell")
    st.stop()

language_model_dir = BASE_MODEL_DIR if base_model_ready else FINETUNED_DIR
tokenizer = load_tokenizer(str(language_model_dir))
language_codes = get_model_language_codes(str(language_model_dir))
language_labels = get_language_labels(language_codes)
code_by_label = {label: code for code, label in language_labels.items()}
language_choices = sorted(code_by_label, key=str.casefold)

sample_sentences = [
    "Clean water is important for health.",
    "Farmers wait for the monsoon rain.",
    "Every child deserves quality education.",
]
selected_sample = st.selectbox("Try a sample sentence", ["Custom input..."] + sample_sentences)
source_text = st.text_area(
    "Text to translate",
    value="" if selected_sample == "Custom input..." else selected_sample,
    height=140,
    placeholder="Type or paste text here",
)

source_mode = st.radio(
    "Source language",
    ["Auto-detect", "Choose manually"],
    horizontal=True,
)
source_language = None
if source_mode == "Auto-detect":
    if source_text.strip():
        detected_code = detect_nllb_language(source_text, language_codes)
        if detected_code:
            source_language = detected_code
            st.info(f"Detected: {language_labels[detected_code]}. Choose manually if this is incorrect.")
        else:
            st.warning("Could not confidently match this text to a supported language. Choose the source language manually.")
else:
    source_label = st.selectbox("Choose the source language", language_choices)
    source_language = code_by_label[source_label]

target_label = st.selectbox("Translate into", ["Choose a target language..."] + language_choices)
target_language = code_by_label.get(target_label)

selected_model_dir = BASE_MODEL_DIR
selected_model_name = "Pretrained multilingual model"
if finetuned_model_ready and source_language == SRC_LANG and target_language == TGT_LANG:
    selected_model_dir = FINETUNED_DIR
    selected_model_name = "Fine-tuned low-resource model"
elif not base_model_ready:
    selected_model_dir = FINETUNED_DIR
    selected_model_name = "Fine-tuned model (base model unavailable)"
if source_language and target_language:
    st.info(f"Using: {selected_model_name}")

if st.button("Translate", type="primary", use_container_width=True):
    if not source_text.strip():
        st.warning("Enter some text to translate.")
    elif source_language is None:
        st.warning("Select the source language manually or enter text that can be detected.")
    elif target_language is None:
        st.warning("Choose a target language.")
    else:
        if not base_model_ready and not (source_language == SRC_LANG and target_language == TGT_LANG):
            st.warning("The pretrained multilingual model is missing; the fine-tuned model may not work well for this language pair.")
        with st.spinner("Loading the local model..."):
            tokenizer, model = load_model(str(selected_model_dir), DEVICE)
        with st.spinner("Translating on this device..."):
            translation = translate(
                source_text.strip(),
                model,
                tokenizer,
                source_language,
                target_language,
                beams,
            )
        st.subheader("Translation")
        st.text_area("Result", value=translation, height=140, label_visibility="collapsed")
        with st.expander("Inspect subword tokens"):
            st.code(" ".join(tokenizer.tokenize(source_text.strip())))
