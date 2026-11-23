from mt_config import BASE_MODEL_DIR, BASE_MODEL_NAME
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {BASE_MODEL_NAME} to {BASE_MODEL_DIR}. This requires internet and several GB of disk space.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    model.save_pretrained(BASE_MODEL_DIR, safe_serialization=True)
    print("Model and tokenizer are ready for offline use.")


if __name__ == "__main__":
    main()
