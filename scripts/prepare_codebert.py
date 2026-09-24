"""Download/cache microsoft/codebert-base for fully local inference.

Run this setup step on a machine with network access. The Streamlit application
uses local_files_only=True after the cache is prepared.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "models" / "huggingface"
MODEL = "microsoft/codebert-base"


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Caching {MODEL} under {CACHE}")
    AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=False)
    AutoModel.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=False)
    print("CodeBERT cache ready. The application will use local_files_only=True.")


if __name__ == "__main__":
    main()
