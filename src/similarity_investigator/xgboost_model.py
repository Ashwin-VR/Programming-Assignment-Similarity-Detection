from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .feature_dataset import NUMERIC_FEATURE_COLUMNS, xgb_matrix_rows, pair_feature_rows
from .models import PairResult

XGBOOST_SCHEMA_VERSION = "xgboost-review-priority-v1"


class XGBoostReviewModel:
    """Local XGBoost classifier for human-labeled review-priority relationships."""

    def __init__(self, model=None, model_path: str | Path | None = None) -> None:
        self._model = model
        self.model_path = Path(model_path) if model_path else None
        self.feature_columns = tuple(NUMERIC_FEATURE_COLUMNS)

    @property
    def available(self) -> bool:
        return self._model is not None

    def _ensure_xgboost(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is not installed. Install requirements.txt before enabling the ML ranker.") from exc
        return XGBClassifier

    def load(self, path: str | Path | None = None) -> None:
        model_path = Path(path or self.model_path or "models/xgboost/review_priority.json")
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        XGBClassifier = self._ensure_xgboost()
        model = XGBClassifier()
        model.load_model(model_path)
        self._model = model
        self.model_path = model_path

    def predict_probability(self, rows: list[dict[str, object]]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("XGBoost model is not loaded")
        matrix = np.asarray(xgb_matrix_rows(rows), dtype=np.float32)
        return self._model.predict_proba(matrix)[:, 1]

    def score_results(self, results: list[PairResult]) -> None:
        if not results:
            return
        rows = pair_feature_rows(results)
        probabilities = self.predict_probability(rows)
        for result, probability in zip(results, probabilities):
            deterministic = float(result.model.get("deterministic_score", result.model.get("score", 0.0)))
            result.model["deterministic_score"] = deterministic
            result.model["score"] = float(np.clip(probability, 0.0, 1.0))
            result.model["model_version"] = XGBOOST_SCHEMA_VERSION
            result.evidence["score_basis"] = "XGBoost review-priority model trained on human-labeled relationship examples"
            result.evidence["ml_probability"] = float(result.model["score"])

    def save(self, path: str | Path) -> Path:
        if not self.available:
            raise RuntimeError("No XGBoost model is loaded")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(destination)
        metadata = destination.with_suffix(destination.suffix + ".metadata.json")
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": XGBOOST_SCHEMA_VERSION,
                    "feature_columns": list(self.feature_columns),
                    "target": "review_priority_label",
                    "target_semantics": "Human-labeled relationship/review outcome. Not a plagiarism verdict.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination


def train_xgboost(
    rows: list[dict[str, object]],
    labels: Iterable[int],
    *,
    seed: int = 42,
    n_estimators: int = 180,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> XGBoostReviewModel:
    labels_array = np.asarray(list(labels), dtype=np.int32)
    if len(rows) != len(labels_array):
        raise ValueError("rows and labels must have the same length")
    if len(rows) < 10:
        raise ValueError("at least 10 labeled pair examples are required")
    if len(np.unique(labels_array)) < 2:
        raise ValueError("training labels must contain both 0 and 1")

    XGBClassifier = XGBoostReviewModel()._ensure_xgboost()
    matrix = np.asarray(xgb_matrix_rows(rows), dtype=np.float32)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=2,
    )
    model.fit(matrix, labels_array)
    return XGBoostReviewModel(model=model)
