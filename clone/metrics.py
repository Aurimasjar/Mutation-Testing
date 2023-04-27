import pandas as pd
from torch import tensor

import prepare_data_java
from prepare_data_c import get_sequences
from prepare_data_java import get_sequence
# from . import prepare_data_java
# from .prepare_data_c import get_sequences
# from .prepare_data_java import get_sequence

class Metrics:
    def __init__(self, *params):
        self.params = params


def get_count(sequence, searched_item):
    count = 0
    for item in sequence:
        if item in searched_item:
            count += 1
    return count


# calculate metrics for c code

# calculate 75 metrics for c code
def calculate_old_c_metrics(ast):
    sequence = []
    get_sequences(ast, sequence)
    if_count = get_count(sequence, ['If'])
    if_true_count = get_count(sequence, ['iftrue'])
    if_false_count = get_count(sequence, ['iffalse'])
    for_count = get_count(sequence, ['For'])
    while_count = get_count(sequence, ['While'])
    do_while_count = get_count(sequence, ['DoWhile'])
    break_count = get_count(sequence, ['Break'])
    continue_count = get_count(sequence, ['Continue'])
    func_call_count = get_count(sequence, ['FuncCall'])
    return_count = get_count(sequence, ['Return'])
    cast_count = get_count(sequence, ['Cast'])
    typename_count = get_count(sequence, ['Typename'])
    decl_count = get_count(sequence, 'Decl')  # Decl, FuncDecl, TypeDecl, PtrDecl, ArrayDecl, ...
    comp_count = get_count(sequence, ['Compound'])
    end_count = get_count(sequence, ['End'])  # count of ending brackets
    ref_count = get_count(sequence, 'Ref')  # ArrayRef, StructRef, ...
    list_count = get_count(sequence, 'List')  # InitList, DeclList, ExprList, ParamList, ...
    constant_count = get_count(sequence, 'Constant')
    id_count = get_count(sequence, 'ID')
    value_count = get_count(sequence, 'value')
    name_count = get_count(sequence, 'name')
    binary_op_count = get_count(sequence, 'BinaryOp')

    char_count = get_count(sequence, ['char'])
    unsigned_char_count = get_count(sequence, ['unsigned char'])
    signed_char_count = get_count(sequence, ['signed char'])
    string_count = get_count(sequence, ['string'])
    int_count = get_count(sequence, ['int'])
    unsigned_int_count = get_count(sequence, ['unsigned int'])
    short_count = get_count(sequence, ['short'])
    unsigned_short_count = get_count(sequence, ['unsigned short'])
    long_count = get_count(sequence, ['long'])
    unsigned_long_count = get_count(sequence, ['unsigned long'])
    float_count = get_count(sequence, ['float'])
    double_count = get_count(sequence, ['double'])
    long_double_count = get_count(sequence, ['long double'])
    bool_count = get_count(sequence, ['bool'])
    void_count = get_count(sequence, ['void'])

    assignment_count = get_count(sequence, ['='])
    all_assignment_count = get_count(sequence, ['Assignment'])
    is_lower_count = get_count(sequence, ['<'])
    is_upper_count = get_count(sequence, ['>'])
    is_lower_or_equal_count = get_count(sequence, ['<='])
    is_upper_or_equal_count = get_count(sequence, ['>='])
    is_equal_count = get_count(sequence, ['=='])
    and_count = get_count(sequence, ['&&'])
    or_count = get_count(sequence, ['||'])
    not_count = get_count(sequence, ['!'])

    plus_count = get_count(sequence, ['+'])
    minus_count = get_count(sequence, ['-'])
    mul_count = get_count(sequence, ['*'])
    div_count = get_count(sequence, ['/'])
    rem_count = get_count(sequence, ['%'])
    addr_count = get_count(sequence, ['&'])
    ternary_op_count = get_count(sequence, ['TernaryOp'])
    increment_count = get_count(sequence, ['++'])
    decrement_count = get_count(sequence, ['--'])

    bitwise_not_count = get_count(sequence, ['~'])
    bitwise_and_count = get_count(sequence, ['&'])
    bitwise_or_count = get_count(sequence, ['|'])
    bitwise_xor_count = get_count(sequence, ['^'])
    bitwise_left_count = get_count(sequence, ['<<'])
    bitwise_right_count = get_count(sequence, ['>>'])

    strlen_count = get_count(sequence, ['strlen'])
    strcpy_count = get_count(sequence, ['strcpy'])
    min_count = get_count(sequence, ['min'])
    max_count = get_count(sequence, ['max'])
    sqrt_count = get_count(sequence, ['sqrt'])
    sizeof_count = get_count(sequence, ['sizeof'])
    malloc_count = get_count(sequence, ['malloc'])
    calloc_count = get_count(sequence, ['calloc'])
    realloc_count = get_count(sequence, ['realloc'])
    free_count = get_count(sequence, ['free'])

    leaf_count = get_leaf_count_c(ast)
    node_count = get_node_count_c(ast)
    max_depth = get_max_depth_c(ast)

    # return [if_count, assignment_count, leaf_count, max_depth]
    return [if_count, if_true_count, if_false_count, for_count, while_count, do_while_count,
            break_count, continue_count, func_call_count, return_count,
            cast_count, typename_count, decl_count, comp_count, end_count, ref_count, list_count, constant_count,
            id_count, value_count, name_count, binary_op_count,
            char_count, unsigned_char_count, signed_char_count, string_count,
            int_count, unsigned_int_count, short_count, unsigned_short_count,
            long_count, unsigned_long_count, float_count, double_count, long_double_count, bool_count, void_count,
            assignment_count, all_assignment_count, is_lower_count, is_upper_count, is_lower_or_equal_count,
            is_upper_or_equal_count, is_equal_count,
            and_count, or_count, not_count, plus_count, minus_count, mul_count, div_count, rem_count,
            addr_count, ternary_op_count, increment_count, decrement_count,
            bitwise_not_count, bitwise_and_count, bitwise_or_count, bitwise_xor_count, bitwise_left_count,
            bitwise_right_count,
            strlen_count, strcpy_count, min_count, max_count, sqrt_count, sizeof_count,
            malloc_count, calloc_count, realloc_count, free_count,
            leaf_count, node_count, max_depth]

# calculate 44 metrics for c code
def calculate_c_metrics(ast):
    sequence = []
    get_sequences(ast, sequence)
    if_count = get_count(sequence, ['If'])
    loop_count = get_count(sequence, ['For', 'While', 'DoWhile'])
    break_continue_count = get_count(sequence, ['Break', 'Continue'])
    func_call_count = get_count(sequence, ['FuncCall'])
    return_count = get_count(sequence, ['Return'])
    cast_count = get_count(sequence, ['Cast'])
    typename_count = get_count(sequence, ['Typename'])
    decl_count = get_count(sequence, 'Decl')  # Decl, FuncDecl, TypeDecl, PtrDecl, ArrayDecl, ...
    comp_count = get_count(sequence, ['Compound'])
    ref_count = get_count(sequence, 'Ref')  # ArrayRef, StructRef, ...
    list_count = get_count(sequence, 'List')  # InitList, DeclList, ExprList, ParamList, ...
    constant_count = get_count(sequence, 'Constant')
    id_count = get_count(sequence, 'ID')
    value_count = get_count(sequence, 'value')
    name_count = get_count(sequence, 'name')

    char_string_count = get_count(sequence, ['char', 'unsigned char', 'signed char', 'string'])
    int_count = get_count(sequence, ['short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long'])
    float_count = get_count(sequence, ['float', 'double', 'long double'])
    bool_count = get_count(sequence, ['bool'])
    void_count = get_count(sequence, ['void'])

    assignment_count = get_count(sequence, ['='])
    all_assignment_count = get_count(sequence, ['Assignment'])
    is_lower_count = get_count(sequence, ['<'])
    is_upper_count = get_count(sequence, ['>'])
    is_lower_or_equal_count = get_count(sequence, ['<='])
    is_upper_or_equal_count = get_count(sequence, ['>='])
    is_equal_count = get_count(sequence, ['=='])
    and_count = get_count(sequence, ['&&'])
    or_count = get_count(sequence, ['||'])
    not_count = get_count(sequence, ['!'])

    plus_count = get_count(sequence, ['+'])
    minus_count = get_count(sequence, ['-'])
    mul_count = get_count(sequence, ['*'])
    div_count = get_count(sequence, ['/'])
    rem_count = get_count(sequence, ['%'])
    addr_count = get_count(sequence, ['&'])
    unary_op_count = get_count(sequence, 'UnaryOp')
    binary_op_count = get_count(sequence, 'BinaryOp')
    ternary_op_count = get_count(sequence, ['TernaryOp'])
    increment_count = get_count(sequence, ['++'])
    decrement_count = get_count(sequence, ['--'])

    leaf_count = get_leaf_count_c(ast)
    node_count = get_node_count_c(ast)
    max_depth = get_max_depth_c(ast)

    # return [if_count, assignment_count, leaf_count, max_depth]
    return [if_count, loop_count, break_continue_count, func_call_count, return_count,
            cast_count, typename_count, decl_count, comp_count, ref_count, list_count, constant_count,
            id_count, value_count, name_count,
            char_string_count, int_count, float_count, bool_count, void_count,
            assignment_count, all_assignment_count, is_lower_count, is_upper_count, is_lower_or_equal_count,
            is_upper_or_equal_count, is_equal_count,
            and_count, or_count, not_count, plus_count, minus_count, mul_count, div_count, rem_count,
            addr_count, unary_op_count, binary_op_count, ternary_op_count, increment_count, decrement_count,
            leaf_count, node_count, max_depth]

# calculate 21 metrics for c code
def calculate_c_metrics_2(ast):
    sequence = []
    get_sequences(ast, sequence)
    if_count = get_count(sequence, ['If'])
    loop_count = get_count(sequence, ['For', 'While', 'DoWhile'])
    break_continue_count = get_count(sequence, ['Break', 'Continue'])
    func_call_count = get_count(sequence, ['FuncCall'])
    return_count = get_count(sequence, ['Return'])
    cast_count = get_count(sequence, ['Cast'])
    decl_count = get_count(sequence, 'Decl')  # Decl, FuncDecl, TypeDecl, PtrDecl, ArrayDecl, ...
    comp_count = get_count(sequence, ['Compound'])
    ref_count = get_count(sequence, 'Ref')  # ArrayRef, StructRef, ...
    list_count = get_count(sequence, 'List')  # InitList, DeclList, ExprList, ParamList, ...
    constant_count = get_count(sequence, 'Constant')

    char_string_count = get_count(sequence, ['char', 'unsigned char', 'signed char', 'string'])
    int_count = get_count(sequence, ['short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long'])
    float_count = get_count(sequence, ['float', 'double', 'long double'])

    all_assignment_count = get_count(sequence, ['Assignment'])
    comparison_count = get_count(sequence, ['<', '>', '<=', '>=', '=='])
    boolean_op_count = get_count(sequence, ['&&', '||', '!'])

    arithmetic_op_count = get_count(sequence, ['+', '-', '*', '/', '%', '++', '--'])

    leaf_count = get_leaf_count_c(ast)
    node_count = get_node_count_c(ast)
    max_depth = get_max_depth_c(ast)

    # return [if_count, assignment_count, leaf_count, max_depth]
    return [if_count, loop_count, break_continue_count, func_call_count, return_count,
            cast_count, decl_count, comp_count, ref_count, list_count, constant_count,
            char_string_count, int_count, float_count,
            all_assignment_count, comparison_count, boolean_op_count, arithmetic_op_count,
            leaf_count, node_count, max_depth]



def get_node_count_c(node):
    node_count = 0
    for _, child in node.children():
        node_count += get_node_count_c(child)
    return node_count + 1


def get_leaf_count_c(node):
    if not node.children():
        return 1
    leaf_count = 0
    for _, child in node.children():
        leaf_count += get_leaf_count_c(child)
    return leaf_count


def get_max_depth_c(node):
    if not node.children():
        return 0
    max_depth = 0
    for _, child in node.children():
        new_depth = get_max_depth_c(child)
        if new_depth > max_depth:
            max_depth = new_depth
    return max_depth + 1


# calculate 44 metrics for java code
def calculate_java_metrics(ast):
    sequence = []
    get_sequence(ast, sequence)
    if_count = get_count(sequence, ['IfStatement'])
    loop_count = get_count(sequence, ['ForStatement', 'WhileStatement', 'DoWhileStatement'])
    break_continue_count = get_count(sequence, ['BreakStatement', 'ContinueStatement'])
    method_inv_count = get_count(sequence, ['MethodInvocation'])
    return_count = get_count(sequence, ['ReturnStatement'])
    cast_count = get_count(sequence, ['Cast'])
    variable_decl_count = get_count(sequence, ['VariableDeclarator'])
    local_variable_decl_count = get_count(sequence, ['LocalVariableDeclaration'])
    final_count = get_count(sequence, ['final'])

    member_ref_count = get_count(sequence, ['MemberReference'])
    ref_type_count = get_count(sequence, ['ReferenceType'])
    list_count = get_count(sequence, ['Collection', 'Queue', 'Deque', 'AbstractSequentialList', 'LinkedList', 'List', 'AbstractList', 'ArrayList', 'NodeList', 'Vector'])
    literal_count = get_count(sequence, ['Literal'])
    null_count = get_count(sequence, ['null'])
    block_count = get_count(sequence, ['BlockStatement'])
    end_count = get_count(sequence, ['End'])

    char_string_count = get_count(sequence, ['char', 'String'])
    int_count = get_count(sequence, ['byte', 'short', 'int', 'long'])
    float_count = get_count(sequence, ['float', 'double'])
    bool_count = get_count(sequence, ['boolean'])
    void_count = get_count(sequence, ['void'])

    assignment_count = get_count(sequence, ['='])
    all_assignment_count = get_count(sequence, ['Assignment'])
    is_lower_count = get_count(sequence, ['<'])
    is_upper_count = get_count(sequence, ['>'])
    is_lower_or_equal_count = get_count(sequence, ['<='])
    is_upper_or_equal_count = get_count(sequence, ['>='])
    is_equal_count = get_count(sequence, ['=='])
    and_count = get_count(sequence, ['&&'])
    or_count = get_count(sequence, ['||'])
    not_count = get_count(sequence, ['!'])

    plus_count = get_count(sequence, ['+'])
    minus_count = get_count(sequence, ['-'])
    mul_count = get_count(sequence, ['*'])
    div_count = get_count(sequence, ['/'])
    rem_count = get_count(sequence, ['%'])
    addr_count = get_count(sequence, ['&'])
    binary_op_count = get_count(sequence, 'BinaryOperation')
    ternary_op_count = get_count(sequence, 'TernaryExpression')
    increment_count = get_count(sequence, ['++'])
    decrement_count = get_count(sequence, ['--'])

    leaf_count = get_leaf_count_java(ast)
    node_count = get_node_count_java(ast)
    max_depth = get_max_depth_java(ast)

    return [if_count, loop_count, break_continue_count, method_inv_count, return_count,
            cast_count, variable_decl_count, local_variable_decl_count, final_count,
            member_ref_count, ref_type_count, list_count,
            literal_count, null_count, block_count, end_count,
            char_string_count, int_count, float_count, bool_count, void_count,
            assignment_count, all_assignment_count,
            is_lower_count, is_upper_count, is_lower_or_equal_count, is_upper_or_equal_count,
            is_equal_count, and_count, or_count, not_count,
            plus_count, minus_count, mul_count, div_count, rem_count,
            addr_count, binary_op_count, ternary_op_count, increment_count, decrement_count,
            leaf_count, node_count, max_depth]


def get_node_count_java(node):
    node_count = 0
    for child in prepare_data_java.get_children(node):
        node_count += get_node_count_java(child)
    return node_count + 1


def get_leaf_count_java(node):
    if not prepare_data_java.get_children(node):
        return 1
    leaf_count = 0
    for child in prepare_data_java.get_children(node):
        leaf_count += get_leaf_count_java(child)
    return leaf_count


def get_max_depth_java(node):
    if not prepare_data_java.get_children(node):
        return 0
    max_depth = 0
    for child in prepare_data_java.get_children(node):
        new_depth = get_max_depth_java(child)
        if new_depth > max_depth:
            max_depth = new_depth
    return max_depth + 1
