from __future__ import annotations
import ast
from .ast_analysis import ast_node_count,ast_signature,canonical_ast,parse_python
from .cfg_analysis import build_cfg_for_language
from .models import StaticFeatures
from .tokens import normalized_tokens
def _definition_counts(tree:ast.AST)->tuple[int,int]:
    return (sum(1 for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))),sum(1 for node in ast.walk(tree) if isinstance(node,ast.ClassDef)))
def _tree_definition_counts(source:str,language:str)->tuple[int,int]:
    from .parsing import parse_tree_sitter
    parsed=parse_tree_sitter(source,language)
    function_types={"cpp":{"function_definition","lambda_expression"},"java":{"method_declaration","constructor_declaration","lambda_expression"}}[language]
    class_types={"cpp":{"class_specifier","struct_specifier"},"java":{"class_declaration","interface_declaration","enum_declaration"}}[language]
    functions=classes=0
    def visit(node):
        nonlocal functions,classes
        if node.type in function_types: functions+=1
        if node.type in class_types: classes+=1
        for child in node.named_children: visit(child)
    visit(parsed.tree.root_node); return functions,classes
def analyze_source(source:str,language:str="python")->StaticFeatures:
    tokens=normalized_tokens(source,language); line_count=len(source.splitlines())
    signature,node_count,parse_error,parse_error_count=canonical_ast(source,language)
    cfg,cfg_error,cfg_error_count=build_cfg_for_language(source,language)
    if language=="python":
        tree=parse_python(source); function_count,class_count=_definition_counts(tree)
    else:
        function_count,class_count=_tree_definition_counts(source,language)
    return StaticFeatures(token_count=len(tokens),normalized_tokens=tokens,ast_node_count=node_count,ast_signature=signature,cfg_nodes=len(cfg.nodes),cfg_edges=len(cfg.edges),cfg_signature=cfg.signature,line_count=line_count,function_count=function_count,class_count=class_count,ast_supported=True,cfg_supported=True,parse_error_count=max(parse_error_count,cfg_error_count),parse_has_error=bool(parse_error or cfg_error))
