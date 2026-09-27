# Code Similarity Investigator

Local, evidence-first investigation tooling for programming assignments.

The application helps a professor inspect measurable relationships between student submissions. It does **not** execute student code, send source code to an external AI service, or make a plagiarism/cheating determination. The final academic judgment remains with the reviewer.

## What is included

The current detector supports exactly three source languages:

- Python
- C++
- Java

Submissions are compared only with other submissions in the same language because the structural detectors are language-specific.

## Tech stack

- **Python 3.10+** for the application and analysis pipeline
- **Streamlit** for the local web dashboard
- **Python `ast`** for Python syntax analysis
- **Tree-sitter** with C++ and Java grammars for structural parsing
- **NumPy** for numeric operations and semantic embeddings
- **NetworkX** for similarity relationship graphs and groups
- **Pandas** for result tables and feature datasets
- **Plotly** for the similarity-space visualization
- **PyTorch + Hugging Face Transformers** for the optional local CodeBERT detector
- **CSV** for the exported pairwise feature dataset

XGBoost is part of the local review-priority pipeline. The current bundled artifact is a synthetic demonstration model. It is explicitly not a plagiarism classifier and must be replaced or revalidated with professor-reviewed relationship labels before production use.

## How it works

```text
Student source files / ZIP
          |
          v
   Secure ingestion
          |
          +--> normalized tokens
          +--> canonical AST
          +--> lightweight CFG
          +--> optional local CodeBERT
          |
          v
   Pairwise evidence features
          |
          +--> common-across-corpus adjustment
          +--> corpus-relative calibration
          |
          v
   XGBoost review-priority model
          |
          v
   Similarity relationships / groups
          |
          v
   Review dashboard + CSV evidence dataset
```

### 1. Secure ingestion

The application accepts individual source files or a ZIP archive. ZIP entries are checked for unsafe paths, traversal attempts, symlink-like entries, nested ZIP files, excessive file counts, and excessive sizes before source files are accepted.

### 2. Token evidence

Source is normalized into language-aware tokens. Token similarity is useful for identifying direct or lightly modified copying and provides interpretable matched regions.

### 3. Structural evidence

Python uses the standard Python AST. C++ and Java use Tree-sitter. The implementation converts the languages into canonical structural signatures that reduce the effect of formatting and identifier renaming. A lightweight control-flow representation captures branches, loops, returns, exceptions, and related control-flow structure.

### 4. Common-code adjustment

Code that appears across a large fraction of the submitted corpus can be downweighted so that shared boilerplate contributes less to the pairwise evidence.

### 5. Optional semantic evidence

The optional CodeBERT detector uses `microsoft/codebert-base` locally. It mean-pools the model representation and compares embeddings with cosine similarity. CodeBERT is an evidence detector here, not a plagiarism classifier.

The application is configured for `local_files_only=True` during investigations. If the model is not already available locally, the semantic detector is skipped and the static detectors continue to work.

### 6. Evidence fusion and calibration

The deterministic, versioned baseline is built from the available static evidence. Results are also calibrated within each language using corpus percentile and a robust MAD-based deviation measure. When the local XGBoost artifact is present, its probability becomes the review-priority score while the deterministic score remains available as `deterministic_score`. The system exposes the underlying detector values rather than turning them into a claim of misconduct.

### 7. Relationship groups

Strong pairwise relationships can be represented as a NetworkX graph. Connected groups help the reviewer inspect clusters of related submissions.

### 8. XGBoost feature dataset and explainability

Every analysis run writes a pairwise CSV containing a fixed numeric feature schema (`xgb-ready-v2`). Missing detector values are represented as missing data at the model boundary. The XGBoost model uses all 44 numeric features in the fixed schema. The dashboard shows global tree gain importance and pair-specific native XGBoost contributions. Pair-specific contributions are additive in model log-odds, so they explain why individual features moved the model output up or down for that pair.

The displayed XGBoost review-priority score is a model output, not an accuracy percentage and not a plagiarism probability. The dashboard also retains the deterministic evidence score and shows global feature importance plus pair-specific native XGBoost contributions so that the reviewer can inspect which measured features moved a model output up or down.

## Project layout

The repository contains the application, analysis modules, setup scripts, and technical documentation needed to reproduce the local website. Generated caches, uploaded data, local model weights, tests, temporary files, and sample student submissions are intentionally kept out of Git.

```text
Programming-Assignment-Similarity-Detection/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
├── SALVO_Recruitment_Task_Documentation_Team-We Love Salvo.pdf
├── app/
│   └── main.py
├── scripts/
│   ├── prepare_codebert.py
│   └── train_demo_xgboost.py
└── src/
    └── similarity_investigator/
        ├── ast_analysis.py
        ├── calibration.py
        ├── cfg_analysis.py
        ├── common_code.py
        ├── feature_dataset.py
        ├── graphs.py
        ├── ingestion.py
        ├── languages.py
        ├── models.py
        ├── pairwise.py
        ├── parsing.py
        ├── relationships.py
        ├── semantic.py
        ├── static_analysis.py
        ├── tokens.py
        └── xgboost_model.py
```

## Complete setup on Windows

This section describes the full setup required by a new user. The application is designed to run locally. No external LLM, plagiarism API, or cloud inference service is required.

### Prerequisites

Install:

1. **Python 3.10 or newer**
2. **Git**
3. PowerShell or another terminal

Check Python and Git:

```powershell
python --version
git --version
```

### Step 1: Clone the repository

```powershell
git clone https://github.com/Ashwin-VR/Programming-Assignment-Similarity-Detection.git
cd Programming-Assignment-Similarity-Detection
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current user, use the appropriate local execution-policy setting for your environment, then activate the environment again.

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install the runtime dependencies

```powershell
python -m pip install -r requirements.txt
```

The dependency list includes the Tree-sitter runtime and the C++ and Java grammars required by the structural detectors.

### Step 5: Verify the source compiles

```powershell
python -m compileall app src
```

A successful command completes without compilation errors.

### Step 6: Prepare the local XGBoost review-priority model

XGBoost is part of the application pipeline. The repository does not store generated model artifacts, so a fresh checkout should create the local demonstration model once:

```powershell
python scripts/train_demo_xgboost.py
```

This creates:

```text
models/xgboost/review_priority.json
```

The demonstration model is trained from synthetic relationship labels. It is included to make the complete ML path runnable, but its output must not be treated as a real-world plagiarism probability or validation result. A production deployment should replace it with a model trained from professor-reviewed relationship outcomes and student-aware validation data.

### Step 7: Optional local CodeBERT setup

CodeBERT adds semantic similarity evidence. It is optional because the token, AST, CFG, common-code, calibration, relationship, and XGBoost pipeline can run without semantic embeddings. To enable it, download the local `microsoft/codebert-base` model once:

```powershell
python scripts/prepare_codebert.py
```

The script stores the model under `models/huggingface`. During an investigation, the application loads CodeBERT with `local_files_only=True`. Student source is not sent to Hugging Face during inference.

If CodeBERT is not prepared, leave **ENABLE LOCAL CODEBERT** unchecked.

### Step 8: Start the application

```powershell
python -m streamlit run app/main.py
```

Streamlit will print the local URL in the terminal. Open that URL in a browser.

### Step 9: Run an investigation

1. Upload individual `.py`, `.cpp`, `.cc`, `.cxx`, `.hpp`, or `.java` files, or upload one ZIP archive.
2. The ZIP can contain a class submission set. Unsafe archive entries are rejected before analysis.
3. Leave **ENABLE LOCAL CODEBERT** unchecked for the simplest setup. Token, AST, CFG, common-code, calibration, relationship, and feature-dataset functionality works without it.
4. If CodeBERT is already installed in the local model cache, enable **ENABLE LOCAL CODEBERT**.
5. Click **PROCESS SUBMISSIONS**.
6. Inspect the pairwise relationships, detector measurements, source comparison, AST, CFG, matched token regions, relationship groups, and exported feature CSV.

## Input limits and security behavior

The current ingestion layer enforces these limits:

- Maximum uploaded archive/file input: **100 MB**
- Maximum individual source file: **5 MB**
- Maximum extracted ZIP contents: **500 MB**
- Maximum files in an archive or submission batch: **500**
- Nested ZIP archives are rejected
- Absolute and traversal paths are rejected
- Symlink-like ZIP entries are rejected
- Unsupported file types are rejected
- Source is parsed as data and **never executed**

These controls are intended to reduce archive traversal, decompression, and unsafe-input risks during local investigation.

## Privacy and locality

The application is designed to run locally. Uploaded student source is processed by the local Streamlit process and local analysis libraries. CodeBERT, when enabled, is loaded with `local_files_only=True` during analysis. No external LLM or plagiarism API is required.

Do not commit student submissions, exported class results, model weights, API keys, PATs, environment files, or other sensitive material to the repository.

## Output and interpretation

The dashboard reports measured similarity relationships and detector evidence. It does not produce a probability of plagiarism, cheating score, or automatic academic-integrity decision.

The main evidence signals are:

- normalized token similarity
- common-code adjusted token similarity
- matched token counts and regions
- canonical AST similarity
- lightweight CFG similarity
- optional CodeBERT semantic similarity
- AST/CFG size ratios
- line, function, and class statistics
- parse-error indicators
- corpus percentile
- robust MAD-based corpus deviation
- language indicators

The final interpretation belongs to the professor or other authorized reviewer.

## Development notes

The bundled XGBoost artifact is a synthetic demonstration model. The next model-stage work should use professor-reviewed relationship outcomes and compare the trained model against the deterministic baseline on a held-out student-aware validation set. Synthetic metrics are not evidence of real-world detector accuracy.

## ML stage: local CodeBERT and XGBoost

The application includes both ML components described in the technical documentation: local CodeBERT for semantic evidence and XGBoost for review-priority scoring. They serve different roles and are not treated as a single plagiarism detector.

### Local CodeBERT

CodeBERT is loaded from the local Hugging Face cache only. The application does not call an external inference API. Run `python scripts/prepare_codebert.py` once on a machine with network access to populate `models/huggingface`, then the application uses `local_files_only=True` for inference. Embeddings are mean-pooled, normalized, and cached under `data/cache/embeddings` using the source hash, model name, and semantic schema version.

The Streamlit option is enabled by default. If the local model is unavailable, the static deterministic evidence pipeline remains available, but a run without CodeBERT does not contain semantic evidence.

### XGBoost review-priority model

XGBoost is the learned fusion stage. The project has a local XGBoost classifier interface in `src/similarity_investigator/xgboost_model.py`. It consumes the exact 44-feature numeric contract from `feature_dataset.py`, including token, AST, CFG, semantic, size, parse, and corpus-relative features. Missing semantic values are passed as `NaN`, which XGBoost can handle. This allows the learned model to assign dynamic weights to the available evidence instead of using only fixed hand-written weights.

A local demonstration model can be trained with:

```text
python scripts/train_demo_xgboost.py
```

This writes:

```text
models/xgboost/review_priority.json
```

The demonstration training labels are synthetic by construction. They prove the end-to-end ML path and are not a substitute for professor-reviewed validation labels. For a defensible final model, replace the demo labels with reviewed relationship outcomes and evaluate the XGBoost model against the deterministic baseline before relying on its review ordering.

When the model artifact exists, the application applies its probability as the review-priority score while preserving the deterministic score in the pair result as `deterministic_score`. The UI also exposes global feature importance and pair-specific native XGBoost contributions. The contribution view is exact for the loaded tree ensemble in log-odds space and is not computed as feature value multiplied by feature importance.
