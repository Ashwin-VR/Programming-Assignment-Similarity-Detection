from __future__ import annotations
import ast
from .cfg_analysis import build_cfg_for_language
from .parsing import parse_tree_sitter
from .ast_analysis import tree_sitter_ast_signature

def _dot_escape(value:str)->str:
    return value.replace("\\","\\\\").replace('"','\\"').replace("\n"," ")

def _ts_ast_dot(source:str,language:str,max_nodes:int=100)->str:
    parsed=parse_tree_sitter(source,language); nodes=[]; parents={}
    def visit(node,parent=None):
        if len(nodes)>=max_nodes or not node.is_named:return
        idx=len(nodes); nodes.append(node)
        if parent is not None: parents[idx]=parent
        for child in node.named_children: visit(child,idx)
    visit(parsed.tree.root_node)
    lines=["digraph AST {","  rankdir=TB;","  node [shape=box];"]
    for i,node in enumerate(nodes): lines.append(f'  n{i} [label="{_dot_escape(node.type)}"];')
    for child,parent in parents.items(): lines.append(f"  n{parent} -> n{child};")
    lines.append("}"); return "\n".join(lines)

def ast_to_dot(source:str,max_nodes:int=80,language:str="python")->str:
    if language=="python":
        tree=ast.parse(source); nodes=[]; parents={}
        def visit(node,parent=None):
            if len(nodes)>=max_nodes:return
            idx=len(nodes);nodes.append(node)
            if parent is not None:parents[idx]=parent
            for child in ast.iter_child_nodes(node): visit(child,idx)
        visit(tree)
        lines=["digraph AST {","  rankdir=TB;","  node [shape=box];"]
        for i,node in enumerate(nodes): lines.append(f'  n{i} [label="{_dot_escape(type(node).__name__)}"];')
        for child,parent in parents.items(): lines.append(f"  n{parent} -> n{child};")
        lines.append("}"); return "\n".join(lines)
    return _ts_ast_dot(source,language,max_nodes)

def cfg_to_dot(source:str,language:str="python")->str:
    graph,_,_=build_cfg_for_language(source,language)
    lines=["digraph CFG {","  rankdir=TB;","  node [shape=box];"]
    for node in graph.nodes: lines.append(f'  n{node.node_id} [label="{_dot_escape(node.label)}"];')
    for source_id,target_id in sorted(graph.edges): lines.append(f"  n{source_id} -> n{target_id};")
    lines.append("}"); return "\n".join(lines)
