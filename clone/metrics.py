import pandas as pd
from torch import tensor

from prepare_data_c import get_sequences
from prepare_data_java import get_sequence
import prepare_data_c
import prepare_data_java


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


def calculate_c_metrics(ast):
    # print('calculate_c_metrics', ast)
    sequence = []
    get_sequences(ast, sequence)
    if_count = get_count(sequence, ['If'])
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

    char_count = get_count(sequence, ['char'])
    unsigned_char_count = get_count(sequence, ['unsigned char'])
    signed_char_count = get_count(sequence, ['signed char'])
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
    is_lower_count = get_count(sequence, ['<'])
    is_upper_count = get_count(sequence, ['>'])
    is_lower_or_equal_count = get_count(sequence, ['<='])
    is_upper_or_equal_count = get_count(sequence, ['>='])
    is_equal_count = get_count(sequence, ['=='])
    plus_count = get_count(sequence, ['+'])
    minus_count = get_count(sequence, ['-'])
    mul_count = get_count(sequence, ['*'])
    div_count = get_count(sequence, ['/'])
    rem_count = get_count(sequence, ['%'])
    addr_count = get_count(sequence, ['&'])
    ternary_op_count = get_count(sequence, ['TernaryOp'])

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
    printf_count = get_count(sequence, ['printf'])
    scanf_count = get_count(sequence, ['scanf'])

    leaf_count = get_leaf_count_c(ast)
    node_count = get_node_count_c(ast)
    max_depth = get_max_depth_c(ast)

    # return [if_count, assignment_count, leaf_count, max_depth]
    return [if_count, for_count, while_count, do_while_count, break_count, continue_count, func_call_count, return_count, cast_count, typename_count, decl_count, comp_count, end_count, ref_count, list_count, char_count, unsigned_char_count, signed_char_count, int_count, unsigned_int_count, short_count, unsigned_short_count, long_count, unsigned_long_count, float_count, double_count, long_double_count, bool_count, void_count, assignment_count, is_lower_count, is_upper_count, is_lower_or_equal_count, is_upper_or_equal_count, is_equal_count, plus_count, minus_count, mul_count, div_count, rem_count, addr_count, ternary_op_count, strlen_count, strcpy_count, min_count, max_count, sqrt_count, sizeof_count, malloc_count, calloc_count, realloc_count, free_count, printf_count, scanf_count, leaf_count, node_count, max_depth]



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


# calculate metrics for java code

def calculate_java_metrics(ast):
    # print('calculate_java_metrics', ast)
    sequence = []
    get_sequence(ast, sequence)
    # print('calculate_java_metrics seq', sequence)
    # todo renew metrics list for java code
    if_count = get_count(sequence, ['IfStatement'])
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

    char_count = get_count(sequence, ['char'])
    unsigned_char_count = get_count(sequence, ['unsigned char'])
    signed_char_count = get_count(sequence, ['signed char'])
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
    is_lower_count = get_count(sequence, ['<'])
    is_upper_count = get_count(sequence, ['>'])
    is_lower_or_equal_count = get_count(sequence, ['<='])
    is_upper_or_equal_count = get_count(sequence, ['>='])
    is_equal_count = get_count(sequence, ['=='])
    plus_count = get_count(sequence, ['+'])
    minus_count = get_count(sequence, ['-'])
    mul_count = get_count(sequence, ['*'])
    div_count = get_count(sequence, ['/'])
    rem_count = get_count(sequence, ['%'])
    addr_count = get_count(sequence, ['&'])
    ternary_op_count = get_count(sequence, ['TernaryOp'])

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
    printf_count = get_count(sequence, ['printf'])
    scanf_count = get_count(sequence, ['scanf'])

    leaf_count = get_leaf_count_java(ast)
    node_count = get_node_count_java(ast)
    max_depth = get_max_depth_java(ast)

    # return Metrics(if_count, assignment_count, leaf_count, max_depth)  # 54 list params, 2 tree params
    # return {'if_count': if_count, 'assignment_count': assignment_count, 'leaf_count': leaf_count,
    #         'max_depth': max_depth}
    return [if_count, assignment_count, leaf_count, max_depth]
    # return pd.Series([if_count, assignment_count, leaf_count, max_depth])



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