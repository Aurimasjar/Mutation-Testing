import pandas as pd
from torch import tensor

from prepare_data import get_sequences


class Metrics:
    def __init__(self, *params):
        self.params = params


def calculate_metrics(ast):
    print('calculate_metrics', ast)
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

    leaf_count = get_leaf_count(ast)
    node_count = get_node_count(ast)
    max_depth = get_max_depth(ast)

    # return Metrics(if_count, assignment_count, leaf_count, max_depth)  # 54 list params, 2 tree params
    # return {'if_count': if_count, 'assignment_count': assignment_count, 'leaf_count': leaf_count,
    #         'max_depth': max_depth}
    return [if_count, assignment_count, leaf_count, max_depth]
    # return pd.Series([if_count, assignment_count, leaf_count, max_depth])


def get_count(sequence, searched_item):
    count = 0
    for item in sequence:
        if item in searched_item:
            count += 1
    return count


def get_node_count(node):
    node_count = 0
    for _, child in node.children():
        node_count += get_node_count(child)
    return node_count + 1


def get_leaf_count(node):
    if not node.children():
        return 1
    leaf_count = 0
    for _, child in node.children():
        leaf_count += get_leaf_count(child)
    return leaf_count


def get_max_depth(node):
    if not node.children():
        return 0
    max_depth = 0
    for _, child in node.children():
        new_depth = get_max_depth(child)
        if new_depth > max_depth:
            max_depth = new_depth
    return max_depth + 1
