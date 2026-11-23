import random

import pandas as pd
import sacrebleu
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from mt_config import BASE_MODEL_DIR, FINETUNED_DIR, ROOT_DIR, SRC_LANG, TGT_LANG

BT_TAG = "<bt>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_SYNTHETIC_RATIO = 1.0


def clean_pairs(pairs):
    cleaned = []
    seen = set()
    for source, target in pairs:
        source = str(source).strip()
        target = str(target).strip()
        pair = (source, target)
        if source and target and pair not in seen:
            cleaned.append(pair)
            seen.add(pair)
    return cleaned


def load_datasets():
    train_path = ROOT_DIR / "data" / "train_parallel.csv"
    val_path = ROOT_DIR / "data" / "val_parallel.csv"
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required parallel corpus file not found: {path}")

    def read_pairs(path):
        frame = pd.read_csv(path)
        missing_columns = {"src", "tgt"} - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} must contain columns named 'src' and 'tgt'.")
        return clean_pairs(frame[["src", "tgt"]].dropna().itertuples(index=False, name=None))

    train_pairs = read_pairs(train_path)
    val_pairs = read_pairs(val_path)
    if not train_pairs:
        raise ValueError(f"No non-empty parallel examples found in {train_path}.")
    if not val_pairs:
        raise ValueError(f"No non-empty parallel examples found in {val_path}.")

    mono_path = ROOT_DIR / "data" / "mono_target.txt"
    mono_lines = []
    if mono_path.is_file():
        mono_lines = list(dict.fromkeys(
            line.strip() for line in mono_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ))
    return train_pairs, val_pairs, mono_lines


def generate_back_translations(model, tokenizer, mono_sentences, batch_size=4):
    if not mono_sentences:
        return []

    tokenizer.src_lang = TGT_LANG
    source_language_id = tokenizer.convert_tokens_to_ids(SRC_LANG)
    if source_language_id is None or source_language_id == tokenizer.unk_token_id:
        raise ValueError(f"{SRC_LANG!r} is not a language code supported by this tokenizer.")

    synthetic_pairs = []
    model.eval()
    for start in range(0, len(mono_sentences), batch_size):
        batch = mono_sentences[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=source_language_id,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        translated_sources = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        synthetic_pairs.extend(
            (f"{BT_TAG} {source.strip()}", target)
            for source, target in zip(translated_sources, batch)
            if source.strip()
        )
    return clean_pairs(synthetic_pairs)


def limit_synthetic_pairs(train_pairs, synthetic_pairs, seed=SEED):
    limit = int(len(train_pairs) * MAX_SYNTHETIC_RATIO)
    if len(synthetic_pairs) <= limit:
        return synthetic_pairs
    return random.Random(seed).sample(synthetic_pairs, limit)


def main():
    if not (BASE_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Offline base model not found at {BASE_MODEL_DIR}. Run `py download_model.py` first."
        )

    set_seed(SEED)
    train_pairs, val_pairs, mono_lines = load_datasets()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR, local_files_only=True).to(DEVICE)
    synthetic_pairs = generate_back_translations(bt_model, tokenizer, mono_lines)
    del bt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    synthetic_pairs = limit_synthetic_pairs(train_pairs, synthetic_pairs)
    combined_train = [{"src": source, "tgt": target} for source, target in train_pairs + synthetic_pairs]
    train_ds = Dataset.from_list(combined_train)
    val_ds = Dataset.from_list([{"src": source, "tgt": target} for source, target in val_pairs])

    def tokenize_batch(batch):
        tokenizer.src_lang = SRC_LANG
        model_inputs = tokenizer(batch["src"], max_length=256, truncation=True)
        tokenizer.tgt_lang = TGT_LANG
        labels = tokenizer(text_target=batch["tgt"], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"])
    tokenized_val = val_ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"])

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    target_language_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if target_language_id is None or target_language_id == tokenizer.unk_token_id:
        raise ValueError(f"{TGT_LANG!r} is not a language code supported by this tokenizer.")
    base_model.config.forced_bos_token_id = target_language_id
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in sequence]
            for sequence in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions = [prediction.strip() for prediction in decoded_predictions]
        references = [label.strip() for label in decoded_labels]
        return {
            "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
            "chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(ROOT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        num_train_epochs=10,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=256,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(
        f"Training on {len(train_pairs)} parallel and {len(synthetic_pairs)} tagged "
        f"back-translated pairs; validating on {len(val_pairs)} pairs."
    )
    trainer.train()
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"Saved fine-tuned model to {FINETUNED_DIR}")


if __name__ == "__main__":
    main()
