# Local Setup (owner checklist)

> Everything lives **inside the project folder** (`D:\healthcare_sentiment`). Tools, model weights and caches go in
> git-ignored sub-folders, so nothing large ends up on `C:` and nothing private ends up in git.

```
D:\healthcare_sentiment\
  .venv\                 Python environment (uv)
  .cache\uv\             uv package cache            (pyproject [tool.uv] cache-dir)
  .cache\huggingface\    Hugging Face models/data    (HF_HOME in .env)
  tools\ollama\          Ollama program              (installer /DIR flag)
  models\ollama\         Ollama model weights        (OLLAMA_MODELS user variable)
  data\raw\…             datasets
  .env                   your keys (never committed)
```

All of `.cache/`, `tools/`, `models/`, `data/` (except `data/annotations/`) and `.env` are git-ignored.

---

## 1. Ollama (needed for Phase 1 pre-annotation and Phase 2)

**Status: installed 2026-10-13** (v0.40.3, `tools\ollama`, ~2.9 GB; `OLLAMA_MODELS` set; added to user PATH by the installer).
Ollama itself always writes a few small files to `C:` that cannot be relocated: `%USERPROFILE%\.ollama` (identity key, small cache) and `%LOCALAPPDATA%\Ollama` (app log/db, ~150 KB). Model weights go to `models\ollama`.

Silent re-install (what the agent ran; installer is ~1.5 GB, signed by Ollama Inc.):
```powershell
.\OllamaSetup.exe /DIR="D:\healthcare_sentiment\tools\ollama" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

Manual steps, for reference:

1. Download **OllamaSetup.exe** from <https://ollama.com/download/windows>.
2. Install it into the project folder. Open PowerShell in the folder where the installer was saved and run:
   ```powershell
   .\OllamaSetup.exe /DIR="D:\healthcare_sentiment\tools\ollama"
   ```
3. Keep model weights in the project folder (they are several GB each):
   - Windows Settings → search **"environment variables"** → **Edit environment variables for your account**
   - **New…** → Name `OLLAMA_MODELS`, Value `D:\healthcare_sentiment\models\ollama` → OK
4. Quit Ollama from the system tray, then start it again from the Start menu (it must restart to see the variable).
5. Check, in a **new** terminal:
   ```powershell
   ollama --version
   ```
   Models are pulled by the agent (Phase 1 pre-annotation: `qwen3.5:9b-q4_K_M`, ~6.6 GB).

**Note (2026-10-13):** the tray app's built-in server timed out at startup (GPU discovery takes ~11 s). The pipeline
starts its own server with the project's model folder instead:
```powershell
$env:OLLAMA_MODELS = "D:\healthcare_sentiment\models\ollama"
Start-Process .\tools\ollama\ollama.exe -ArgumentList serve -WindowStyle Hidden
```

## 2. Hugging Face account + token (needed in Phase 3 to publish models)

1. Sign up at <https://huggingface.co/join>.
2. <https://huggingface.co/settings/tokens> → **Create new token** → type **Write** → copy it.
3. Paste it into `.env` (next to the project's `.env.example`) as `HF_TOKEN=hf_…`.

## 3. openFDA API key (needed in Phase 4; free)

1. Go to <https://open.fda.gov/apis/authentication/> and request a key with your email.
2. Paste it into `.env` as `OPENFDA_API_KEY=…`.

## 4. WebMD dataset (needed for the cross-platform test)

1. Sign in to Kaggle and open <https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset>.
2. **Download**, unzip, and put the CSV at `D:\healthcare_sentiment\data\raw\webmd\webmd.csv`.
3. Verify:
   ```bash
   uv run python -m medsignal.data.download --only webmd
   ```
   It should log `webmd: manual files present`.

## 5. PsyTAR download (one minute)

The PsyTAR source is bot-protected, so scripts cannot fetch it. In your browser open
<https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp>,
download the dataset (xlsx) and save it as `D:\healthcare_sentiment\data\raw\psytar\PsyTAR_dataset.xlsx`.
Check with `uv run python -m medsignal.data.download --only psytar`.

## 5b. (Parked) Annotating a Drugs.com gold set

> Not required: replaced by expert corpora (ADR-0018). Kept for an optional in-domain spot check.

The agent prepares everything (gold sample, LLM pre-annotations for dev, task files, interface). You annotate.

1. **Start Label Studio** (data stays in `.cache\label-studio`), in PowerShell from the project folder:
   ```powershell
   $env:LABEL_STUDIO_BASE_DATA_DIR = "D:\healthcare_sentiment\.cache\label-studio"
   .\tools\label-studio\Scripts\label-studio.exe start --port 8080
   ```
   It opens <http://localhost:8080>. First time: **Sign up** (a local account stored only on this PC).
2. **Create project "MedSignal gold - dev"**
   - *Labeling Setup* → *Code* → paste the contents of `configs\labelstudio.xml` → Save.
   - *Data Import* → upload `data\interim\labelstudio\tasks_dev.json`.
   - *Settings → Annotation* → turn on **Use predictions to prefill annotations**.
3. **Annotate** following `docs/annotation_guidelines.md`: select ADE text with the *ADE* label, click the span to set severity / flags, answer *Discontinued* and *Effectiveness*, then **Submit**. Pre-filled answers are suggestions: check every field.
4. **Export**: *Export* → **JSON** → save as `data\interim\labelstudio\export_dev.json`. Then tell the agent, or run:
   ```bash
   uv run python -m medsignal.labeling.labelstudio import --export data/interim/labelstudio/export_dev.json --split dev
   ```
5. Repeat with project **"MedSignal gold - test"** using `tasks_test.json` (no pre-fill: test is annotated blind, ADR-0017) and `--split test`.

Tips: take breaks every ~50 reviews; use *notes* for anything unclear; you can export at any time (re-importing replaces the file).

## 6. Optional cleanup of legacy folders

These are left over from the old project. Both are git-ignored and nothing in the new pipeline reads them:

| Folder | Size | Status |
|---|---|---|
| `healthcare_sentiment\` | ~263 MB | Drugs.com file already copied to `data\raw\drugscom_uci\`; the Russian facility set is unused |
| `healthcare_bert_final\` | ~256 MB | old DistilBERT weights (results withdrawn) |

Delete them only if you don't want to keep a personal backup; the old code is preserved at git tag `v0-minor-project`.

---

When a step is done, tick it in issue #2 on GitHub (or tell the agent, who will update `docs/STATUS.md`).
