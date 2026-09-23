from __future__ import annotations
from collections import Counter
from math import ceil
from .models import StaticFeatures
from .tokens import token_similarity

def common_tokens(features:dict[str,StaticFeatures],min_document_fraction:float=0.5)->set[str]:
    if not features:return set()
    threshold=max(2,ceil(len(features)*min_document_fraction))
    counts=Counter()
    for item in features.values(): counts.update(set(item.normalized_tokens))
    return {token for token,count in counts.items() if count>=threshold and token not in {"<NAME>","<NUM>","<STR>"}}

def common_token_fraction(tokens:tuple[str,...],common:set[str])->float:
    if not tokens:return 0.0
    return sum(token in common for token in tokens)/len(tokens)

def adjusted_token_similarity(left:tuple[str,...],right:tuple[str,...],common:set[str])->tuple[float,float,float]:
    left_fraction=common_token_fraction(left,common); right_fraction=common_token_fraction(right,common)
    if not common:return token_similarity(left,right),left_fraction,right_fraction
    left_filtered=tuple(token for token in left if token not in common)
    right_filtered=tuple(token for token in right if token not in common)
    if not left_filtered and not right_filtered:return token_similarity(left,right),left_fraction,right_fraction
    return token_similarity(left_filtered,right_filtered),left_fraction,right_fraction
