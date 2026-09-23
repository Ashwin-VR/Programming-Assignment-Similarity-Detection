from __future__ import annotations
import ast
import difflib
from dataclasses import dataclass,field
@dataclass(frozen=True)
class CFGNode:
    node_id:int
    label:str
@dataclass
class CFG:
    nodes:list[CFGNode]=field(default_factory=list)
    edges:set[tuple[int,int]]=field(default_factory=set)
    @property
    def signature(self)->tuple[str,...]:
        return tuple(node.label for node in self.nodes)
class _CFGBuilder:
    def __init__(self)->None:
        self.graph,self._next_id=CFG(),0
    def node(self,label:str)->int:
        node_id=self._next_id; self._next_id+=1; self.graph.nodes.append(CFGNode(node_id,label)); return node_id
    def connect(self,source:int|None,target:int)->None:
        if source is not None: self.graph.edges.add((source,target))
    def block(self,statements:list[ast.stmt],incoming:int|None=None)->list[int]:
        exits=[incoming] if incoming is not None else [None]
        for statement in statements: exits=self.statement(statement,exits)
        return [item for item in exits if item is not None]
    def statement(self,statement:ast.stmt,incoming:list[int|None])->list[int]:
        current=self.node(_python_cfg_label(type(statement).__name__))
        for source in incoming: self.connect(source,current)
        if isinstance(statement,ast.If):
            body_exits=self.block(statement.body,current); else_exits=self.block(statement.orelse,current) if statement.orelse else [current]
            join=self.node("Join")
            for source in body_exits+else_exits: self.connect(source,join)
            return [join]
        if isinstance(statement,(ast.For,ast.AsyncFor,ast.While)):
            body_exits=self.block(statement.body,current)
            for source in body_exits: self.connect(source,current)
            exits=[current]
            if statement.orelse: exits.extend(self.block(statement.orelse,current))
            join=self.node("LoopJoin")
            for source in exits: self.connect(source,join)
            return [join]
        if isinstance(statement,ast.Try):
            exits=self.block(statement.body,current)
            for handler in statement.handlers: exits.extend(self.block(handler.body,current))
            if statement.orelse: exits.extend(self.block(statement.orelse,current))
            join=self.node("TryJoin")
            for source in exits: self.connect(source,join)
            if statement.finalbody: return self.block(statement.finalbody,join)
            return [join]
        if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            self.block(statement.body,current); return [current]
        if isinstance(statement,(ast.Return,ast.Raise,ast.Break,ast.Continue)): return []
        return [current]

def _python_cfg_label(kind:str)->str:
    return {
        "FunctionDef":"FUNCTION","AsyncFunctionDef":"FUNCTION","ClassDef":"CLASS",
        "If":"IF","For":"FOR","AsyncFor":"FOR","While":"WHILE","Try":"TRY",
        "With":"WITH","Return":"RETURN","Raise":"THROW","Break":"BREAK",
        "Continue":"CONTINUE","Join":"JOIN","LoopJoin":"LOOP_JOIN","TryJoin":"TRY_JOIN",
    }.get(kind, "STMT")

def build_cfg(source:str)->CFG:
    builder=_CFGBuilder(); builder.block(ast.parse(source).body); return builder.graph
def _bounded_count_similarity(left:int,right:int)->float:
    maximum=max(left,right); return 1.0 if maximum==0 else max(0.0,1.0-abs(left-right)/maximum)
def cfg_similarity_from_features(left_signature,left_nodes,left_edges,right_signature,right_nodes,right_edges)->float:
    sequence_score=difflib.SequenceMatcher(a=left_signature,b=right_signature,autojunk=False).ratio()
    return max(0.0,min(1.0,0.6*sequence_score+0.2*_bounded_count_similarity(left_nodes,right_nodes)+0.2*_bounded_count_similarity(left_edges,right_edges)))
_TS_STATEMENTS={"expression_statement","declaration","local_variable_declaration","field_declaration","return_statement","throw_statement","break_statement","continue_statement","goto_statement","assert_statement","empty_statement","labeled_statement"}
class _TreeSitterCFGBuilder:
    def __init__(self)->None:
        self.graph,self._next_id=CFG(),0
    def node(self,label:str)->int:
        node_id=self._next_id; self._next_id+=1; self.graph.nodes.append(CFGNode(node_id,label)); return node_id
    def connect(self,source:int|None,target:int)->None:
        if source is not None: self.graph.edges.add((source,target))
    def field(self,node,*names):
        for name in names:
            child=node.child_by_field_name(name)
            if child is not None: return child
        return None
    def block_children(self,node):
        return [child for child in node.named_children if child.type!="comment"]
    def simple(self,node,incoming):
        current=self.node(_cfg_label(node.type))
        for source in incoming: self.connect(source,current)
        if node.type in {"return_statement","throw_statement","break_statement","continue_statement","goto_statement"}: return []
        return [current]
    def process_sequence(self,children,incoming):
        exits=list(incoming)
        for child in children: exits=self.process(child,exits)
        return exits
    def process(self,node,incoming):
        if node is None: return incoming
        kind=node.type
        if kind in {"compound_statement","block","statement_block","class_body"}: return self.process_sequence(self.block_children(node),incoming)
        if kind in {"function_definition","method_definition","method_declaration","constructor_declaration","class_specifier","class_declaration","interface_declaration","enum_declaration","struct_specifier"}:
            current=self.node("FUNCTION")
            for source in incoming: self.connect(source,current)
            body=self.field(node,"body")
            return self.process(body,[current]) if body is not None else [current]
        if kind=="if_statement":
            branch=self.node("IF")
            for source in incoming: self.connect(source,branch)
            consequence=self.field(node,"consequence","body"); alternative=self.field(node,"alternative","else")
            body_exits=self.process(consequence,[branch]); alt_exits=self.process(alternative,[branch]) if alternative is not None else [branch]
            join=self.node("JOIN")
            for source in body_exits+alt_exits: self.connect(source,join)
            return [join]
        if kind in {"for_statement","enhanced_for_statement","while_statement","do_statement"}:
            label={"for_statement":"FOR","enhanced_for_statement":"FOR","while_statement":"WHILE","do_statement":"DO_WHILE"}[kind]
            loop=self.node(label)
            for source in incoming: self.connect(source,loop)
            body=self.field(node,"body"); body_exits=self.process(body,[loop])
            for source in body_exits: self.connect(source,loop)
            join=self.node("LOOP_JOIN"); self.connect(loop,join); return [join]
        if kind in {"switch_statement","switch_expression"}:
            switch=self.node("SWITCH")
            for source in incoming: self.connect(source,switch)
            body=self.field(node,"body"); exits=self.process_sequence(self.block_children(body),[switch]) if body is not None else [switch]
            join=self.node("SWITCH_JOIN")
            for source in exits: self.connect(source,join)
            self.connect(switch,join); return [join]
        if kind in {"try_statement","try_with_resources_statement"}:
            current=self.node("TRY")
            for source in incoming: self.connect(source,current)
            body=self.field(node,"body"); exits=self.process(body,[current])
            for child in node.named_children:
                if child.type=="catch_clause": exits.extend(self.process(child,[current]))
            finally_node=self.field(node,"finally")
            if finally_node is not None: exits=self.process(finally_node,exits)
            join=self.node("TRY_JOIN")
            for source in exits: self.connect(source,join)
            return [join]
        if kind=="catch_clause":
            current=self.node("CATCH")
            for source in incoming: self.connect(source,current)
            return self.process(self.field(node,"body"),[current])
        if kind=="finally_clause":
            current=self.node("FINALLY")
            for source in incoming: self.connect(source,current)
            return self.process(self.field(node,"body"),[current])
        if kind in {"case_statement","switch_block_statement_group"}:
            current=self.node("CASE")
            for source in incoming: self.connect(source,current)
            return self.process_sequence(self.block_children(node),[current])
        if kind in _TS_STATEMENTS: return self.simple(node,incoming)
        structural={"class_specifier","class_declaration","interface_declaration","enum_declaration","struct_specifier"}
        if kind in structural:
            current=self.node("CLASS")
            for source in incoming: self.connect(source,current)
            return self.process(self.field(node,"body"),[current])
        nested_types=_TS_STATEMENTS|{"compound_statement","block","statement_block","class_body","if_statement","for_statement","enhanced_for_statement","while_statement","do_statement","switch_statement","switch_expression","try_statement","try_with_resources_statement","catch_clause","finally_clause","case_statement","switch_block_statement_group","function_definition","method_definition","method_declaration","constructor_declaration","class_specifier","class_declaration","interface_declaration","enum_declaration","struct_specifier"}
        nested=[child for child in node.named_children if child.type in nested_types]
        return self.process_sequence(nested,incoming) if nested else incoming
def _cfg_label(kind:str)->str:
    return {"expression_statement":"EXPR","declaration":"DECL","local_variable_declaration":"DECL","field_declaration":"DECL","return_statement":"RETURN","throw_statement":"THROW","break_statement":"BREAK","continue_statement":"CONTINUE","goto_statement":"GOTO","assert_statement":"ASSERT","empty_statement":"NOOP","labeled_statement":"LABEL"}.get(kind,kind.upper())
def build_tree_sitter_cfg(parsed):
    builder=_TreeSitterCFGBuilder(); builder.process(parsed.tree.root_node,[]); return builder.graph
def build_cfg_for_language(source:str,language:str):
    if language=="python": return build_cfg(source),False,0
    from .parsing import parse_tree_sitter
    parsed=parse_tree_sitter(source,language); return build_tree_sitter_cfg(parsed),parsed.has_error,parsed.error_count
def cfg_similarity(left:str|CFG,right:str|CFG)->float:
    left_graph=build_cfg(left) if isinstance(left,str) else left; right_graph=build_cfg(right) if isinstance(right,str) else right
    return cfg_similarity_from_features(left_graph.signature,len(left_graph.nodes),len(left_graph.edges),right_graph.signature,len(right_graph.nodes),len(right_graph.edges))
