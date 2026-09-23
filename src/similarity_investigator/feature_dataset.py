from __future__ import annotations
import csv
from pathlib import Path
from typing import Final
from .models import PairResult

FEATURE_SCHEMA_VERSION: Final[str] = "xgb-ready-v2"
IDENTIFIER_COLUMNS: tuple[str,...] = ("student_a","student_b","language")
NUMERIC_FEATURE_COLUMNS: tuple[str,...] = (
    "language_code","is_python","is_java","is_cpp",
    "token_similarity","common_adjusted_token_similarity","common_token_fraction_a","common_token_fraction_b",
    "ast_similarity","cfg_similarity","semantic_similarity",
    "matched_token_count","matched_region_count","shared_subtree_count",
    "left_token_count","right_token_count","token_count_ratio",
    "left_ast_node_count","right_ast_node_count","ast_node_count_ratio",
    "left_cfg_nodes","right_cfg_nodes","cfg_node_count_ratio",
    "left_cfg_edges","right_cfg_edges","cfg_edge_count_ratio",
    "left_line_count","right_line_count","line_count_ratio",
    "left_function_count","right_function_count","function_count_difference","function_count_ratio",
    "left_class_count","right_class_count","class_count_difference","class_count_ratio",
    "file_size_ratio","parse_error_count_a","parse_error_count_b","parse_has_error_a","parse_has_error_b",
    "corpus_percentile","corpus_mad_z",
)
FEATURE_COLUMNS: tuple[str,...] = IDENTIFIER_COLUMNS + NUMERIC_FEATURE_COLUMNS

def feature_schema() -> dict[str,object]:
    return {
        "schema_version":FEATURE_SCHEMA_VERSION,
        "identifier_columns":list(IDENTIFIER_COLUMNS),
        "numeric_feature_columns":list(NUMERIC_FEATURE_COLUMNS),
        "numeric_dtype":"float32/float64 compatible",
        "missing_numeric_value":"NaN at model boundary",
        "target":"review_priority_label",
        "target_semantics":"Human-labeled relationship/review outcome. The detector never generates the target.",
        "xgboost_contract":"Build the matrix using NUMERIC_FEATURE_COLUMNS in exactly this order. Keep identifiers outside the matrix.",
    }

def pair_feature_rows(results:list[PairResult])->list[dict[str,object]]:
    rows=[]
    for result in results:
        row={"student_a":result.student_a,"student_b":result.student_b,"language":result.evidence.get("language","Unknown")}
        row.update({key:result.features.get(key) for key in NUMERIC_FEATURE_COLUMNS})
        rows.append(row)
    return rows

def validate_feature_rows(rows:list[dict[str,object]])->None:
    expected=set(FEATURE_COLUMNS)
    for row in rows:
        if set(row)!=expected:
            raise ValueError(f"feature schema mismatch: missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}")
        for column in NUMERIC_FEATURE_COLUMNS:
            value=row[column]
            if value is not None and not isinstance(value,(int,float)):
                raise TypeError(f"{column} must be numeric or None, got {type(value).__name__}")

def xgb_matrix_rows(rows:list[dict[str,object]])->list[list[float]]:
    validate_feature_rows(rows)
    matrix=[]
    for row in rows:
        matrix.append([float("nan") if row[column] is None else float(row[column]) for column in NUMERIC_FEATURE_COLUMNS])
    return matrix

def save_pair_features(results:list[PairResult],path:str|Path)->Path:
    destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True)
    rows=pair_feature_rows(results); validate_feature_rows(rows)
    with destination.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(FEATURE_COLUMNS)); writer.writeheader(); writer.writerows(rows)
    return destination
