from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from .languages import language_key
try:
    from tree_sitter import Language, Parser
except ImportError:
    Language = None
    Parser = None

@dataclass(frozen=True)
class ParsedProgram:
    language: str
    tree: Any
    has_error: bool
    error_count: int
    source_bytes: bytes

def _tree_sitter_language(key: str):
    if Language is None or Parser is None:
        raise RuntimeError("Tree-sitter dependencies are not installed. Install requirements.txt before analyzing Java or C++.")
    if key == "cpp":
        import tree_sitter_cpp
        return Language(tree_sitter_cpp.language())
    if key == "java":
        import tree_sitter_java
        return Language(tree_sitter_java.language())
    raise ValueError(f"Tree-sitter is not configured for {key!r}")

@lru_cache(maxsize=4)
def _parser(key: str):
    return Parser(_tree_sitter_language(key))

def parse_tree_sitter(source: str, language: str) -> ParsedProgram:
    key = language_key(language) if language in {"Python","C++","Java"} else language
    if key not in {"cpp","java"}:
        raise ValueError(f"Tree-sitter parser is only used for C++ and Java, got {language!r}")
    data = source.encode("utf-8")
    tree = _parser(key).parse(data)
    errors = 0
    def walk(node) -> None:
        nonlocal errors
        if node.is_error or node.is_missing:
            errors += 1
        for child in node.children:
            walk(child)
    walk(tree.root_node)
    return ParsedProgram(key, tree, errors > 0, errors, data)
