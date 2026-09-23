from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
@dataclass(frozen=True)
class Submission:
    student_id:str
    filename:str
    source:str
    source_hash:str
    language:str="python"
@dataclass(frozen=True)
class StaticFeatures:
    token_count:int
    normalized_tokens:tuple[str,...]
    ast_node_count:int|None
    ast_signature:tuple[str,...]
    cfg_nodes:int|None
    cfg_edges:int|None
    cfg_signature:tuple[str,...]
    line_count:int
    function_count:int
    class_count:int
    ast_supported:bool=True
    cfg_supported:bool=True
    parse_error_count:int=0
    parse_has_error:bool=False
@dataclass
class PairResult:
    student_a:str
    student_b:str
    features:dict[str,object]=field(default_factory=dict)
    model:dict[str,str|float]=field(default_factory=dict)
    evidence:dict[str,object]=field(default_factory=dict)
@dataclass(frozen=True)
class InputFile:
    name:str
    data:bytes
def source_path_name(path:str|Path)->str: return Path(path).name
