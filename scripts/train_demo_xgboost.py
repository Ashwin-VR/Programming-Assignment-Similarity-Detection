"""Train a local demonstration XGBoost model from synthetic labeled relationships.

This creates a usable ML artifact for the local demo. The labels are synthetic by
construction and must not be treated as real-world detector validation. Replace
this training data with professor-reviewed labels before relying on the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from similarity_investigator.models import Submission
from similarity_investigator.pairwise import compare_all
from similarity_investigator.feature_dataset import pair_feature_rows
from similarity_investigator.xgboost_model import train_xgboost


def python_programs() -> list[str]:
    return [
        "def calc_{i}(values):\n    total = 0\n    for value in values:\n        total += value * {k}\n    return total\n",
        "def choose_{i}(value):\n    if value < {k}:\n        return value + 1\n    return value - 1\n",
        "def count_{i}(values):\n    result = 0\n    for value in values:\n        if value % {k} == 0:\n            result += 1\n    return result\n",
        "def average_{i}(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n",
        "def find_{i}(values, target):\n    for index, value in enumerate(values):\n        if value == target:\n            return index\n    return -1\n",
        "def transform_{i}(values):\n    return [value * {k} + 1 for value in values]\n",
        "def clamp_{i}(value):\n    if value < 0:\n        return 0\n    if value > {k}:\n        return {k}\n    return value\n",
        "def words_{i}(items):\n    result = {}\n    for item in items:\n        result[item] = result.get(item, 0) + 1\n    return result\n",
    ]


def cpp_programs() -> list[str]:
    return [
        "int calc_{i}(int a, int b) {{ return (a + b) * {k}; }}\n",
        "int choose_{i}(int value) {{ if (value < {k}) return value + 1; return value - 1; }}\n",
        "int count_{i}(const std::vector<int>& values) {{ int result = 0; for (int value : values) if (value % {k} == 0) ++result; return result; }}\n",
        "double average_{i}(const std::vector<int>& values) {{ if (values.empty()) return 0; int total = 0; for (int value : values) total += value; return static_cast<double>(total) / values.size(); }}\n",
        "int find_{i}(const std::vector<int>& values, int target) {{ for (size_t index = 0; index < values.size(); ++index) if (values[index] == target) return static_cast<int>(index); return -1; }}\n",
        "int clamp_{i}(int value) {{ if (value < 0) return 0; if (value > {k}) return {k}; return value; }}\n",
    ]


def java_programs() -> list[str]:
    return [
        "class Demo{i} {{ static int calc(int a, int b) {{ return (a + b) * {k}; }} }}\n",
        "class Demo{i} {{ static int choose(int value) {{ if (value < {k}) return value + 1; return value - 1; }} }}\n",
        "class Demo{i} {{ static int count(int[] values) {{ int result = 0; for (int value : values) if (value % {k} == 0) result++; return result; }} }}\n",
        "class Demo{i} {{ static double average(int[] values) {{ if (values.length == 0) return 0; int total = 0; for (int value : values) total += value; return (double) total / values.length; }} }}\n",
        "class Demo{i} {{ static int find(int[] values, int target) {{ for (int index = 0; index < values.length; index++) if (values[index] == target) return index; return -1; }} }}\n",
        "class Demo{i} {{ static int clamp(int value) {{ if (value < 0) return 0; if (value > {k}) return {k}; return value; }} }}\n",
    ]


def build_submissions() -> tuple[list[Submission], dict[tuple[str, str], int]]:
    submissions: list[Submission] = []
    labels: dict[tuple[str, str], int] = {}
    counter = 0
    for language, templates in (("Python", python_programs()), ("C++", cpp_programs()), ("Java", java_programs())):
        for i, template in enumerate(templates):
            base = template.format(i=i, k=(i % 4) + 2)
            variant = template.format(i=i + 100, k=(i % 4) + 2)
            # A structural-preserving formatting/name variant is a synthetic positive.
            a = f"demo_{counter}_a"; counter += 1
            b = f"demo_{counter}_b"; counter += 1
            submissions.extend([
                Submission(a, f"{a}.txt", base, a, language),
                Submission(b, f"{b}.txt", variant, b, language),
            ])
            labels[(a, b)] = 1
            labels[(b, a)] = 1
    # Same-language pairs from different templates are synthetic unrelated examples.
    for i, left in enumerate(submissions):
        for right in submissions[i + 1:]:
            if left.language == right.language and (left.student_id, right.student_id) not in labels:
                labels[(left.student_id, right.student_id)] = 0
    return submissions, labels


def main() -> None:
    submissions, label_map = build_submissions()
    results = compare_all(submissions)
    rows = pair_feature_rows(results)
    labels = [label_map[(r.student_a, r.student_b)] for r in results]
    model = train_xgboost(rows, labels, seed=42)
    destination = ROOT / "models" / "xgboost" / "review_priority.json"
    model.save(destination)
    print(f"trained {len(rows)} synthetic pair examples")
    print(f"positive labels: {sum(labels)}")
    print(f"negative labels: {len(labels) - sum(labels)}")
    print(f"model: {destination}")


if __name__ == "__main__":
    main()
