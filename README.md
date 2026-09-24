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

XGBoost is **not** part of the current decision pipeline. The application currently produces an XGBoost-ready feature schema so that model validation can be performed later with human-labeled data.

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

The current review score is a deterministic, versioned baseline built from the available static evidence. Results are also calibrated within each language using corpus percentile and a robust MAD-based deviation measure. The system exposes the underlying detector values rather than turning them into a claim of misconduct.

### 7. Relationship groups

Strong pairwise relationships can be represented as a NetworkX graph. Connected groups help the reviewer inspect clusters of related submissions.

### 8. XGBoost-ready dataset

Every analysis run writes a pairwise CSV containing a fixed numeric feature schema (`xgb-ready-v2`). Missing detector values are represented as missing data at the model boundary. No XGBoost model is trained or used yet. A future model must be validated against human-labeled review data and compared with the deterministic baseline before it is introduced.

## Project layout

Only the runtime files required to set up and run the application are part of the distributable repository:

```text
Code-Similarity-Investigator/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
├── app/
│   └── main.py
└── src/
    └── similarity_investigator/
        ├── __init__.py
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
        └── tokens.py
```

Generated caches, uploaded data, local model weights, tests, temporary files, and sample student submissions are intentionally not part of the repository.

## Setup on Windows

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

### Step 6: Start the application

```powershell
python -m streamlit run app/main.py
```

Streamlit will print the local URL in the terminal. Open that URL in a browser.

### Step 7: Run an investigation

1. Upload individual `.py`, `.cpp`, `.cc`, `.cxx`, `.hpp`, or `.java` files, or upload one ZIP archive.
2. The ZIP can contain a class submission set. Unsafe archive entries are rejected before analysis.
3. Leave **ENABLE LOCAL CODEBERT** unchecked for the simplest setup. Token, AST, CFG, common-code, calibration, relationship, and feature-dataset functionality works without it.
4. If CodeBERT is already installed in the local model cache, enable **ENABLE LOCAL CODEBERT**.
5. Click **PROCESS SUBMISSIONS**.
6. Inspect the pairwise relationships, detector measurements, source comparison, AST, CFG, matched token regions, relationship groups, and exported feature CSV.

## Optional: enable local CodeBERT

CodeBERT is optional. The application expects the `microsoft/codebert-base` tokenizer and model to be available in the local Hugging Face cache under the project's `models/huggingface` directory.

The repository does **not** include model weights because they are large generated artifacts. Download the model once on a machine with network access, using the same `cache_dir` shown below, then keep the resulting model files local:

```powershell
python -c "from transformers import AutoTokenizer, AutoModel; p='models/huggingface'; AutoTokenizer.from_pretrained('microsoft/codebert-base', cache_dir=p, local_files_only=False); AutoModel.from_pretrained('microsoft/codebert-base', cache_dir=p, local_files_only=False)"
```

After the weights are present, the investigation itself uses local-only model loading. Student source code is not sent to Hugging Face or another external service by this application.

If CodeBERT is unavailable, the dashboard continues without semantic similarity.

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

The current repository intentionally stops before the XGBoost stage. The next model-stage work should use a labeled validation corpus and compare the trained model against the deterministic baseline. If the model does not improve review prioritization in validation, the deterministic baseline should remain the production path.

## ML stage: local CodeBERT and XGBoost

The application now includes both required ML components.

### Local CodeBERT

CodeBERT is loaded from the local Hugging Face cache only. The application does not call an external inference API. Place `microsoft/codebert-base` in `models/huggingface` or populate that cache from a machine with network access before running the semantic detector. Embeddings are mean-pooled, normalized, and cached under `data/cache/embeddings` using the source hash, model name, and semantic schema version.

The Streamlit option is enabled by default. If the local model is unavailable, the static deterministic evidence pipeline remains available, but a run without CodeBERT does not contain semantic evidence.

### XGBoost review-priority model

The project now has a local XGBoost classifier interface in `src/similarity_investigator/xgboost_model.py`. It consumes the exact numeric feature contract from `feature_dataset.py`, including token, AST, CFG, semantic, size, parse, and corpus-relative features. Missing semantic values are passed as `NaN`, which XGBoost can handle.

A local demonstration model can be trained with:

```text
python scripts/train_demo_xgboost.py
```

This writes:

```text
models/xgboost/review_priority.json
```

The demonstration training labels are synthetic by construction. They prove the end-to-end ML path and are not a substitute for professor-reviewed validation labels. For a defensible final model, replace the demo labels with reviewed relationship outcomes and evaluate the XGBoost model against the deterministic baseline before relying on its review ordering.

When the model artifact exists, the application applies its probability as the review-priority score while preserving the deterministic score in the pair result as `deterministic_score`. The UI continues to expose the underlying detector measurements.
