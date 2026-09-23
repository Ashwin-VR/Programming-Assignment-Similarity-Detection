from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SEMANTIC_SCHEMA_VERSION = "codebert-meanpool-v1"


def embedding_cache_key(source: str, model_name: str, schema_version: str) -> str:
    payload = f"{model_name}\n{schema_version}\n{source}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    value = float(np.dot(left, right) / (left_norm * right_norm))
    return max(-1.0, min(1.0, value))


class EmbeddingCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return np.load(path)

    def put(self, key: str, embedding: np.ndarray) -> None:
        path = self.path_for(key)
        np.save(path, embedding.astype(np.float32))


class CodeBERTEncoder:
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        cache_dir: str | Path = "models/huggingface",
        embedding_cache_dir: str | Path = "data/cache/embeddings",
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache = EmbeddingCache(embedding_cache_dir)
        self.local_files_only = local_files_only
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model.eval()

    def embed(self, source: str) -> np.ndarray:
        key = embedding_cache_key(source, self.model_name, SEMANTIC_SCHEMA_VERSION)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded = self._tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with self._torch.no_grad():
            output = self._model(**encoded)

        hidden = output.last_hidden_state.detach().cpu().numpy()
        mask = encoded["attention_mask"].detach().cpu().numpy()
        pooled = mean_pool(hidden, mask)[0]

        norm = np.linalg.norm(pooled)
        embedding = pooled / norm if norm else pooled
        self.cache.put(key, embedding)
        return embedding.astype(np.float32)

    def similarity(self, left_source: str, right_source: str) -> float:
        return cosine_similarity(self.embed(left_source), self.embed(right_source))
