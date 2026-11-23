import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BASE_MODEL_NAME = os.getenv("MT_BASE_MODEL_NAME", "facebook/nllb-200-distilled-600M")
BASE_MODEL_DIR = Path(os.getenv("MT_BASE_MODEL_DIR", str(ROOT_DIR / "models" / "nllb-200-distilled-600M")))
FINETUNED_DIR = Path(os.getenv("MT_FINETUNED_DIR", str(ROOT_DIR / "mt_lora_model")))
SRC_LANG = os.getenv("MT_SRC_LANG", "eng_Latn")
TGT_LANG = os.getenv("MT_TGT_LANG", "hin_Deva")
