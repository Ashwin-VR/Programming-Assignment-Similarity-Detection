from __future__ import annotations
from pathlib import PurePosixPath
SUPPORTED_LANGUAGES={".py":"Python",".cpp":"C++",".cc":"C++",".cxx":"C++",".hpp":"C++",".java":"Java"}
LANGUAGE_ALIASES={"Python":"python","python":"python","C++":"cpp","cpp":"cpp","Java":"java","java":"java"}
LANGUAGE_CODES={"python":0,"java":1,"cpp":2}
def detect_language(filename:str)->str:
    suffix=PurePosixPath(filename).suffix.lower()
    try:return SUPPORTED_LANGUAGES[suffix]
    except KeyError as exc:raise ValueError(f"unsupported source type: {filename!r}") from exc
def language_key(language:str)->str:
    try:return LANGUAGE_ALIASES[language]
    except KeyError as exc:raise ValueError(f"unsupported language: {language!r}") from exc
def source_mode(language:str)->str:return language_key(language)
