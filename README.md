# Offline low-resource translator

A local Streamlit app for translating between the 200+ language/script variants supported by NLLB-200. It can detect the source language offline, lets users override the detection, and asks them to choose a target language. The app does not call a translation API: after the model and dependencies have been installed, inference and language detection work without internet. It adapts the pretrained multilingual model with LoRA and can use target-language monolingual text for tagged back-translation.

## Install and prepare the model

Use Python 3.10 or newer. The pretrained NLLB-200 distilled 600M model needs several GB of disk space and substantial RAM; a CUDA-capable GPU makes training and translation faster.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
```

The one-time model download requires internet. Keep the generated `models\nllb-200-distilled-600M` directory on the computer that will host the app. Once it is present, model loading and translation use local files only.

## Run on a laptop and phone

Run the app on the hosting computer:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

On the laptop, open `http://localhost:8501`. For a phone, connect it to the same Wi-Fi/local network, find the laptop's IPv4 address with `ipconfig`, and open `http://<laptop-ip>:8501` in the phone browser. The layout adapts to narrow screens. Keep the hosting computer and app running while translating from the phone; translation is computed on that computer. This setup needs no internet once installed, but it is not a standalone mobile app and the phone cannot translate when disconnected from the host or local network. Do not expose the unauthenticated Streamlit server to an untrusted or public network.

## Fine-tune for a low-resource language pair

Create a `data` directory with:

- `train_parallel.csv` and `val_parallel.csv`, each with `src` and `tgt` columns.
- Optional `mono_target.txt`, one target-language sentence per line.

The app's language selectors use the language codes present in the installed NLLB tokenizer. Automatic source detection uses the offline `langid` package; short or unsupported text can be ambiguous, so users can choose the source language manually. The training script still defaults to `eng_Latn` and `hin_Deva`; set `MT_SRC_LANG` and `MT_TGT_LANG` for another training pair. `MT_BASE_MODEL_NAME`, `MT_BASE_MODEL_DIR`, and `MT_FINETUNED_DIR` can override model paths; set the same values when downloading, training, and running the app.

```powershell
python train_mt.py
```

The training pipeline uses multilingual transfer learning, tagged stochastic back-translation from target-side monolingual text, and LoRA to update a small set of attention parameters. It cleans blank and duplicate pairs, caps synthetic examples at the size of the real parallel corpus to limit noise, uses dynamic batch padding and gradient accumulation, selects the best checkpoint by validation chrF, and reports BLEU and chrF. Back-translation is skipped when no monolingual file is provided. Training uses only the local base model and local metric implementations; it does not fetch models or evaluation scripts from the internet.

After fine-tuning, restart the app to use the fine-tuned model. If it is absent, the app uses the pretrained model. The selected model's weights are not bundled with this source code.

Run the language detection tests with `py -m unittest discover -s tests`.
