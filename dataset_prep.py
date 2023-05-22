import os
import pathlib
import re

import numpy as np
import pandas as pd

from OriginalMethod import OriginalMethod
from mutant import Mutant

session_path = 'dataset_session'
algorithm_class = 'DatasetRootAlgorithm'
test_file = 'dataset_output.txt'


def print_mutants(mutants):
    for mutant in mutants:
        print(mutant.__dict__)


def get_mutants(original_methods):
    mutants = []
    for p in pathlib.Path('mujava/' + session_path + '/result/' + algorithm_class + '/traditional_mutants').glob(
            '**/*.java'):
        method = pathlib.PurePath(p).parent.parent.name
        operator = pathlib.PurePath(p).parent.name
        method_code = get_method_code(p.read_text(), method)
        # method_original_code = get_method_code(original_code, method)  # todo optimize to parse methods in advance
        mutants.append(Mutant(p.name, method, operator, method_code))
    return mutants


def get_original_methods():
    original_code = pathlib.Path(
        'mujava/' + session_path + '/result/' + algorithm_class + '/original/DatasetRootAlgorithm.java').read_text()
    methods = pathlib.Path(
        'mujava/' + session_path + '/result/' + algorithm_class + '/traditional_mutants/method_list').read_text().split(
        '\n')[:-1]
    original_methods = []
    for method in methods:
        original_methods.append(OriginalMethod(method, get_method_code(original_code, method)))
    return original_methods


def get_method_code(code, method):
    method_name = ' ' + method.split('(')[0].split('_')[1] + '('
    # method_name = method_name.replace('_', ' ')
    code_line_list = code.split('\n')
    pos1 = [idx for idx, s in enumerate(code_line_list) if method_name in s][0]
    pos2 = get_method_end_position(code_line_list, pos1)
    assert pos2 != -1
    return '\n'.join(code_line_list[pos1:pos2+1])


def get_method_end_position(code_line_list, pos1):
    depth = 1
    for i in range(pos1 + 2, len(code_line_list)):
        if '{' in code_line_list[i] and '}' in code_line_list[i]:
            continue
        elif '{' in code_line_list[i]:
            depth += 1
        elif '}' in code_line_list[i]:
            depth -= 1
        if depth == 0:
            return i
    return -1


# classify mutants using data from file, generated with mujava tool
# in the source file it must be ensured that all methods passed written tests, otherwise mutant test results are not reliable
def get_mutant_classification_map():
    mutant_map = {}
    fp = open('./mujava/' + session_path + '/' + test_file, 'r')
    test_report = [line.rstrip() for line in fp]
    fp.close()
    index = test_report.index('======================================== Generating Original Test Results '
                              '========================================')
    test_case_count = test_report[index + 1].count('pass')
    for idx, line in enumerate(test_report):
        if line.startswith('  '):
            operator = line.split('{')[0].split(' ')
            if 'time_out:' in operator:
                operator = operator[operator.index('time_out:') - 1]
            else:
                operator = operator[-1]
            is_equivalent = line.count('pass') == test_case_count
            mutant_map[operator] = is_equivalent

    return mutant_map


def classify_mutants(mutants):
    mutant_map = get_mutant_classification_map()
    mutants = list(filter(lambda m : m.operator in mutant_map, mutants))
    for mutant in mutants:
        mutant.set_is_equivalent(mutant_map[mutant.operator])
    return mutants


def form_dataset(original_methods, mutants):
    original_method_codes = list(map(lambda m: m.code, original_methods))
    mut_dataset = pd.DataFrame(original_method_codes, columns=['code'])
    mut_dataset.reset_index(inplace=True)
    mut_pairs = pd.DataFrame([], columns=['id1', 'id2', 'label'])
    for mutant in mutants:
        index = len(mut_dataset.index)
        mut_dataset.loc[index] = [index, mutant.method_code]
        # omindex = list(filter(lambda m: m.method == mutant.method, original_methods))[0]
        omindex = [i for i in range(len(original_methods)) if original_methods[i].method == mutant.method]
        equiv = 0 if mutant.is_equivalent else 1
        new_pair = pd.DataFrame({'id1': index, 'id2': omindex, 'label': mutant.is_equivalent})
        mut_pairs = pd.concat([mut_pairs, new_pair], ignore_index=True)

    mut_dataset.to_csv('clone/data/javamut/mut_funcs_all.csv', index=False)
    mut_pairs.to_csv('clone/data/javamut/mut_pair_ids.csv')


def prepare_dataset():
    original_methods = get_original_methods()
    print('original methods collected')
    mutants = get_mutants(original_methods)
    print('mutants collected')
    mutants = classify_mutants(mutants)
    print('mutants classified')
    # print_mutants(mutants)
    form_dataset(original_methods, mutants)
    print('mutant dataset formed')


def main():
    print('main')
    prepare_dataset()
    print('main finished')


if __name__ == "__main__":
    main()
