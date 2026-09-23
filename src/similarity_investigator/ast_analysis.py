from __future__ import annotations
import ast
import difflib
from collections import Counter
from collections.abc import Iterable
from .parsing import parse_tree_sitter

_PYTHON_CANONICAL = {
    "Module":"PROGRAM","FunctionDef":"FUNCTION","AsyncFunctionDef":"FUNCTION","ClassDef":"CLASS",
    "arguments":"PARAMETERS","arg":"PARAMETER","If":"IF","For":"FOR","AsyncFor":"FOR","While":"WHILE",
    "Try":"TRY","ExceptHandler":"CATCH","With":"WITH","Return":"RETURN","Break":"BREAK","Continue":"CONTINUE",
    "Raise":"THROW","Assign":"ASSIGN","AnnAssign":"ASSIGN","AugAssign":"ASSIGN","NamedExpr":"ASSIGN",
    "Call":"CALL","BinOp":"BINARY_OP","UnaryOp":"UNARY_OP","BoolOp":"BOOLEAN_OP","Compare":"COMPARE",
    "Constant":"LITERAL","Name":"IDENTIFIER","Attribute":"MEMBER_ACCESS","Subscript":"INDEX",
    "List":"COLLECTION","Tuple":"COLLECTION","Set":"COLLECTION","Dict":"COLLECTION","Expr":"EXPR",
    "Lambda":"LAMBDA","ListComp":"COMPREHENSION","SetComp":"COMPREHENSION","DictComp":"COMPREHENSION",
    "GeneratorExp":"COMPREHENSION","Pass":"NOOP",
}
_TS_CANONICAL = {
    "function_definition":"FUNCTION","method_definition":"FUNCTION","method_declaration":"FUNCTION",
    "constructor_declaration":"FUNCTION","class_specifier":"CLASS","class_declaration":"CLASS",
    "interface_declaration":"CLASS","enum_declaration":"CLASS","struct_specifier":"CLASS",
    "parameter_declaration":"PARAMETER","formal_parameter":"PARAMETER","formal_parameters":"PARAMETERS",
    "parameter_list":"PARAMETERS","argument_list":"ARGUMENTS","compound_statement":"BLOCK","block":"BLOCK",
    "statement_block":"BLOCK","if_statement":"IF","else_clause":"ELSE","for_statement":"FOR",
    "enhanced_for_statement":"FOR","while_statement":"WHILE","do_statement":"DO_WHILE",
    "switch_statement":"SWITCH","switch_expression":"SWITCH","case_statement":"CASE",
    "switch_block_statement_group":"CASE","try_statement":"TRY","try_with_resources_statement":"TRY",
    "catch_clause":"CATCH","finally_clause":"FINALLY","return_statement":"RETURN","break_statement":"BREAK",
    "continue_statement":"CONTINUE","throw_statement":"THROW","goto_statement":"GOTO","call_expression":"CALL",
    "method_invocation":"CALL","assignment_expression":"ASSIGN","variable_declarator":"ASSIGN",
    "local_variable_declaration":"DECL","field_declaration":"DECL","declaration":"DECL",
    "expression_statement":"EXPR","binary_expression":"BINARY_OP","unary_expression":"UNARY_OP",
    "update_expression":"UPDATE","conditional_expression":"TERNARY","subscript_expression":"INDEX",
    "array_access":"INDEX","field_expression":"MEMBER_ACCESS","field_access":"MEMBER_ACCESS",
    "member_expression":"MEMBER_ACCESS","identifier":"IDENTIFIER","field_identifier":"IDENTIFIER",
    "type_identifier":"TYPE","scoped_identifier":"IDENTIFIER","property_identifier":"IDENTIFIER",
    "integer_literal":"LITERAL","floating_point_literal":"LITERAL","decimal_integer_literal":"LITERAL",
    "hex_integer_literal":"LITERAL","string_literal":"LITERAL","character_literal":"LITERAL",
    "true":"LITERAL","false":"LITERAL","null_literal":"LITERAL","object_creation_expression":"NEW",
    "new_expression":"NEW","array_creation_expression":"NEW","lambda_expression":"LAMBDA","comment":"COMMENT",
}

def parse_python(source: str) -> ast.AST:
    return ast.parse(source)

def _python_signature(node: ast.AST, output: list[str]) -> None:
    label = _PYTHON_CANONICAL.get(type(node).__name__, type(node).__name__.upper())
    output.append(label)
    for child in ast.iter_child_nodes(node):
        _python_signature(child, output)
    output.append(f"/{label}")

def ast_signature(tree: ast.AST) -> tuple[str,...]:
    output: list[str] = []
    _python_signature(tree, output)
    return tuple(output)

def _ts_label(node) -> str:
    return _TS_CANONICAL.get(node.type, node.type.upper())

def tree_sitter_ast_signature(parsed) -> tuple[str,...]:
    output: list[str] = []
    def visit(node) -> None:
        if not node.is_named:
            return
        label = _ts_label(node)
        if label == "COMMENT":
            return
        output.append(label)
        for child in node.named_children:
            visit(child)
        output.append(f"/{label}")
    visit(parsed.tree.root_node)
    return tuple(output)

def ast_node_count(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))

def tree_sitter_node_count(parsed) -> int:
    count = 0
    def visit(node) -> None:
        nonlocal count
        if node.is_named: count += 1
        for child in node.named_children: visit(child)
    visit(parsed.tree.root_node)
    return count

def canonical_ast(source: str, language: str) -> tuple[tuple[str,...],int,bool,int]:
    if language == "python":
        tree = parse_python(source)
        return ast_signature(tree), ast_node_count(tree), False, 0
    parsed = parse_tree_sitter(source, language)
    return tree_sitter_ast_signature(parsed), tree_sitter_node_count(parsed), parsed.has_error, parsed.error_count

def signature_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_sig,right_sig = tuple(left),tuple(right)
    if not left_sig and not right_sig: return 1.0
    if not left_sig or not right_sig: return 0.0
    return difflib.SequenceMatcher(a=left_sig,b=right_sig,autojunk=False).ratio()

def shared_subtree_count(left: Iterable[str], right: Iterable[str], minimum_size: int = 2) -> int:
    return sum(1 for block in difflib.SequenceMatcher(a=tuple(left),b=tuple(right),autojunk=False).get_matching_blocks() if block.size >= minimum_size)

def ast_fingerprint(signature: Iterable[str]) -> dict[str,int]:
    return dict(Counter(signature))

def ast_similarity(left: str|ast.AST, right: str|ast.AST) -> float:
    left_tree = parse_python(left) if isinstance(left,str) else left
    right_tree = parse_python(right) if isinstance(right,str) else right
    return signature_similarity(ast_signature(left_tree),ast_signature(right_tree))
