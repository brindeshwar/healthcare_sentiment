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
   Do **not** pull models yet; we benchmark candidates first (Phase 2 picks which ones).

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

## 5. Optional cleanup of legacy folders

These are left over from the old project. Both are git-ignored and nothing in the new pipeline reads them:

| Folder | Size | Status |
|---|---|---|
| `healthcare_sentiment\` | ~263 MB | Drugs.com file already copied to `data\raw\drugscom_uci\`; the Russian facility set is unused |
| `healthcare_bert_final\` | ~256 MB | old DistilBERT weights (results withdrawn) |

Delete them only if you don't want to keep a personal backup; the old code is preserved at git tag `v0-minor-project`.

---

When a step is done, tick it in issue #2 on GitHub (or tell the agent, who will update `docs/STATUS.md`).
