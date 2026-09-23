from __future__ import annotations
import difflib
import math
from collections.abc import Iterable
from itertools import combinations
from .ast_analysis import signature_similarity,shared_subtree_count
from .calibration import calibrate_results
from .common_code import adjusted_token_similarity,common_tokens
from .languages import language_key
from .models import PairResult,StaticFeatures,Submission
from .static_analysis import analyze_source

def cosine_similarity(left, right) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

def _ratio(left:int|None,right:int|None)->float|None:
    if left is None or right is None:return None
    if max(left,right)==0:return 1.0
    return min(left,right)/max(left,right)

def _matching_token_evidence(left:tuple[str,...],right:tuple[str,...])->tuple[int,int,list[dict[str,int]]]:
    blocks=[block for block in difflib.SequenceMatcher(a=left,b=right,autojunk=False).get_matching_blocks() if block.size]
    return sum(block.size for block in blocks),len(blocks),[{"left_start":b.a,"right_start":b.b,"size":b.size} for b in blocks]

def _compare_features(left_submission,right_submission,left,right,semantic_similarity_value=None,common:set[str]|None=None)->PairResult:
    common=common or set()
    matched_count,region_count,regions=_matching_token_evidence(left.normalized_tokens,right.normalized_tokens)
    raw_token=__import__("similarity_investigator.tokens",fromlist=["token_similarity"]).token_similarity(left.normalized_tokens,right.normalized_tokens)
    adjusted_token,left_common_fraction,right_common_fraction=adjusted_token_similarity(left.normalized_tokens,right.normalized_tokens,common)
    ast_score=signature_similarity(left.ast_signature,right.ast_signature) if left.ast_supported and right.ast_supported else None
    ast_blocks=[]
    if ast_score is not None:
        ast_blocks=[{"left_start":b.a,"right_start":b.b,"size":b.size} for b in difflib.SequenceMatcher(a=left.ast_signature,b=right.ast_signature,autojunk=False).get_matching_blocks() if b.size >= 2]
    cfg_score=__import__("similarity_investigator.cfg_analysis",fromlist=["cfg_similarity_from_features"]).cfg_similarity_from_features(left.cfg_signature,left.cfg_nodes or 0,left.cfg_edges or 0,right.cfg_signature,right.cfg_nodes or 0,right.cfg_edges or 0) if left.cfg_supported and right.cfg_supported else None
    language=left_submission.language
    language_key_value=language_key(language)
    if ast_score is not None and cfg_score is not None:
        static_score=max(0.0,min(1.0,0.40*adjusted_token+0.40*ast_score+0.20*cfg_score))
        basis="adjusted token + AST + CFG"
    else:
        static_score=adjusted_token; basis="adjusted normalized token evidence"
    features={
        "token_similarity":raw_token,"common_adjusted_token_similarity":adjusted_token,
        "common_token_fraction_a":left_common_fraction,"common_token_fraction_b":right_common_fraction,
        "ast_similarity":ast_score,"cfg_similarity":cfg_score,"semantic_similarity":semantic_similarity_value,
        "matched_token_count":matched_count,"matched_region_count":region_count,
        "shared_subtree_count":shared_subtree_count(left.ast_signature,right.ast_signature),
        "left_token_count":left.token_count,"right_token_count":right.token_count,"token_count_ratio":_ratio(left.token_count,right.token_count),
        "left_ast_node_count":left.ast_node_count,"right_ast_node_count":right.ast_node_count,"ast_node_count_ratio":_ratio(left.ast_node_count,right.ast_node_count),
        "left_cfg_nodes":left.cfg_nodes,"right_cfg_nodes":right.cfg_nodes,"cfg_node_count_ratio":_ratio(left.cfg_nodes,right.cfg_nodes),
        "left_cfg_edges":left.cfg_edges,"right_cfg_edges":right.cfg_edges,"cfg_edge_count_ratio":_ratio(left.cfg_edges,right.cfg_edges),
        "left_line_count":left.line_count,"right_line_count":right.line_count,"line_count_ratio":_ratio(left.line_count,right.line_count),
        "left_function_count":left.function_count,"right_function_count":right.function_count,"function_count_difference":abs(left.function_count-right.function_count),"function_count_ratio":_ratio(left.function_count,right.function_count),
        "left_class_count":left.class_count,"right_class_count":right.class_count,"class_count_difference":abs(left.class_count-right.class_count),"class_count_ratio":_ratio(left.class_count,right.class_count),
        "file_size_ratio":_ratio(len(left_submission.source.encode()),len(right_submission.source.encode())),
        "parse_error_count_a":left.parse_error_count,"parse_error_count_b":right.parse_error_count,
        "parse_has_error_a":int(left.parse_has_error),"parse_has_error_b":int(right.parse_has_error),
        "language_code": {"python":0,"java":1,"cpp":2}[language_key_value],
        "is_python":int(language_key_value=="python"),"is_java":int(language_key_value=="java"),"is_cpp":int(language_key_value=="cpp"),
        "corpus_percentile":None,"corpus_mad_z":None,
    }
    explanation=[f"Language: {language}.","Token similarity measured on normalized source tokens.","Common-across-corpus token regions are downweighted before the deterministic baseline score."]
    explanation.append("AST similarity measured on canonical syntax structure." if ast_score is not None else "AST similarity unavailable.")
    explanation.append("CFG similarity measured on lightweight control-flow structure." if cfg_score is not None else "CFG similarity unavailable.")
    explanation.append("Semantic similarity measured with local CodeBERT mean-pooled embeddings." if semantic_similarity_value is not None else "Semantic similarity unavailable until the local CodeBERT model is installed.")
    return PairResult(left_submission.student_id,right_submission.student_id,features,{"score":static_score,"model_version":"deterministic-baseline-v3"},{"matched_token_regions":regions,"explanation":explanation,"score_basis":basis,"language":language})

def compare_submissions(left:Submission,right:Submission,semantic_encoder=None)->PairResult:
    if language_key(left.language)!=language_key(right.language): raise ValueError("cross-language comparison is disabled")
    key=language_key(left.language)
    semantic=semantic_encoder.similarity(left.source,right.source) if semantic_encoder is not None else None
    return _compare_features(left,right,analyze_source(left.source,key),analyze_source(right.source,key),semantic)

def compare_all(submissions:Iterable[Submission],semantic_encoder=None,progress_callback=None)->list[PairResult]:
    items=list(submissions); features={}
    for index,item in enumerate(items,1):
        key=language_key(item.language); features[item.student_id]=analyze_source(item.source,key)
        if progress_callback: progress_callback("features",index,len(items))
    common=common_tokens(features)
    embeddings={}
    if semantic_encoder is not None:
        for index,item in enumerate(items,1):
            embeddings[item.student_id]=semantic_encoder.embed(item.source)
            if progress_callback: progress_callback("semantic",index,len(items))
    candidates=[pair for pair in combinations(items,2) if language_key(pair[0].language)==language_key(pair[1].language)]
    results=[]
    for pair_index,(left,right) in enumerate(candidates,1):
        semantic=None
        if semantic_encoder is not None: semantic=cosine_similarity(embeddings[left.student_id],embeddings[right.student_id])
        results.append(_compare_features(left,right,features[left.student_id],features[right.student_id],semantic,common))
        if progress_callback: progress_callback("pairs",pair_index,len(candidates))
    calibrate_results(results)
    return results
