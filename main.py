import ast
import random
import sys
from ast import literal_eval
import os
import threading
from itertools import islice
from pathlib import Path
import jpype
import numpy as np
from jpype import java

import pathlib

import javalang
import pandas as pd

import mutants_prep
import plot
from clone.eval import evaluate
from constants import absolute_path
from genetic_algorithm import mutate, crossover, generate_bit_input
from mutant import Mutant
from test_case import TestCase


def print_mutant_list(mutants):
    print('length of mutant list', len(mutants))
    # for i in range(0, len(mutants)):
    #     print('mutant', i, mutants[i].filename, mutants[i].method, mutants[i].operator)


def apply_mutation_testing(return_type, method_name, method_params):
    print('apply_mutation_testing')

    test_set_data = pd.read_csv(
        'test_sets/' + form_method_signature(return_type, method_name, method_params) + 'test_set.csv'
    )
    # print('test set data')
    # print(test_set_data)
    test_set = []
    for test_case in test_set_data.values:
        for i in range(0, len(test_case)):
            if isinstance(test_case[i], str):
                test_case[i] = ast.literal_eval(test_case[i])
        test_set.append(TestCase(test_case, 8))

    mutants = get_mutants(method_name)
    mutants = list(
        filter(lambda m: m.method == form_method_signature(return_type, method_name, method_params), mutants))
    # print_mutant_list(mutants)
    verify_initial_test_set(method_name, test_set)
    scores = [evaluate_mutation_score(method_name, mutants, test_set)]

    mutants = mark_equivalent_mutants(method_name, mutants)
    print_mutant_list(mutants)
    if len(mutants) == 0:
        print('All mutants were killed... Stopping mutation testing process...')
        exit()

    scores.append(evaluate_mutation_score(method_name, mutants, test_set))
    print('scores', scores)


def apply_mutation_testing_with_test_data_generation(return_type, method_name, method_params, set_initial_data=False):
    print('apply_mutation_testing_with_test_data_generation...')
    test_set = []
    if set_initial_data:
        test_set_data = pd.read_csv(
            'test_sets/' + form_method_signature(return_type, method_name, method_params) + 'test_set.csv'
        )
        for test_case in test_set_data.values:
            for i in range(0, len(test_case)):
                if isinstance(test_case[i], str):
                    test_case[i] = ast.literal_eval(test_case[i])
            test_set.append(TestCase(test_case, 8))
    else:
        test_set = generate_initial_test_cases(method_params, method_name, 3)

    mutants = get_mutants(method_name)
    mutants = list(
        filter(lambda m: m.method == form_method_signature(return_type, method_name, method_params), mutants))
    # print_mutant_list(mutants)
    verify_initial_test_set(method_name, test_set)
    print('initial test set verified')
    mutants = mark_equivalent_mutants(method_name, mutants)
    print_mutant_list(mutants)
    if len(mutants) == 0:
        print('All mutants were killed... Stopping mutation testing process...')
        exit()

    test_set = convert_to_bits(test_set)
    all_score, non_eq_score, non_eq_mutants_ratio = evaluate_mutation_score(method_name, mutants, test_set, True)
    all_scores = [all_score]
    non_eq_scores = [non_eq_score]
    non_eq_mutants_ratios = [non_eq_mutants_ratio]
    GA_ITERATION_COUNT = 200
    for i in range(0, GA_ITERATION_COUNT):
        if i % 20 == 0:
            print('genetic algorithm iteration', i)
        apply_genetic_operations(test_set, method_name, method_params)
        test_set = convert_from_bits(test_set)
        recalculate_outputs(test_set, method_name)
        all_score, non_eq_score, non_eq_mutants_ratio = evaluate_mutation_score(method_name, mutants, test_set,
                                                                                i + 1 == GA_ITERATION_COUNT)
        all_scores.append(all_score)
        non_eq_scores.append(non_eq_score)
        non_eq_mutants_ratios.append(non_eq_mutants_ratio)
    # print('scores', all_scores, non_eq_scores, non_eq_mutants_ratio)
    # print score graph
    if set_initial_data:
        plot.plot_mutation_score_ga(all_scores, non_eq_scores, non_eq_mutants_ratios, 'plot_triangle_ga_with_initial_test_set')
    else:
        plot.plot_mutation_score_ga(all_scores, non_eq_scores, non_eq_mutants_ratios, 'plot_triangle_ga_with_random_test_set')

def generate_initial_test_cases(method_params, method_name, num_of_test_cases):
    test_set = []
    Algorithm = jpype.JClass('mujava.program_session.result.Algorithm.original.Algorithm')
    alg = Algorithm()
    for i in range(0, num_of_test_cases):
        test_case = []
        for j in range(0, len(method_params)):
            test_case.append(random.randrange(0, pow(2, 8)))
        test_case.append(getattr(alg, method_name)(*test_case))
        test_set.append(TestCase(test_case, 8))
    return test_set


def convert_to_bits(test_set):
    for test_case in test_set:
        test_case.convert_to_bits()
    return test_set


def convert_from_bits(test_set):
    for test_case in test_set:
        test_case.convert_from_bits()
    return test_set


def recalculate_outputs(test_set, method_name):
    for test_case in test_set:
        test_case.recalculate_output(method_name)
    return test_set


def apply_genetic_operations(test_set, method_name, method_params):
    # print('apply_genetic_operations')
    removal_threshold = 0.2
    threshold = 0.5
    amount_of_test_cases = len(test_set)

    # selection
    for test_case in test_set:
        if test_case.mutant_proportion <= removal_threshold:
            # print('remove', test_case.bit_input)
            test_set.remove(test_case)

    # mutation
    for test_case in test_set:
        if test_case.mutant_proportion <= threshold:
            # print('mutate', test_case.bit_input)
            while True:
                new_test_case = mutate(test_case.bit_input)
                if len(list(filter(lambda x: x.bit_input == new_test_case, test_set))) == 0:
                    test_case.bit_input = new_test_case
                    break

    # new data generation if less than 2 test cases are alive
    while len(test_set) < 2:
        new_test_case = generate_bit_input(8 * len(method_params))
        # print('new test case', new_test_case)
        if len(list(filter(lambda x: x.bit_input == new_test_case, test_set))) == 0:
            test_set.append(TestCase(None, 8, new_test_case, method_name))

    # crossover
    while len(test_set) < amount_of_test_cases:
        new_test_case = crossover(test_set)
        # print('new crossover test case, amount_of_test_cases', new_test_case, amount_of_test_cases)
        if len(list(filter(lambda x: x.bit_input == new_test_case, test_set))) == 0:
            test_set.append(TestCase(None, 8, new_test_case, method_name))


def form_method_signature(return_type, method_name, method_params):
    return return_type + '_' + method_name + '_' + '_'.join(method_params) + '_'


def evaluate_mutation_score(method_name, mutants, test_set, print_table=False):
    # print('evaluate_mutation_score')
    considered_non_eq_mutants_length = len(list(filter(lambda m: not m.is_equivalent or m.is_killed, mutants)))
    # print('(length of mutant list, marked equivalent mutants length)', len(mutants), considered_non_eq_mutants_length)
    inverted_mutation_table = []
    marked_eq_mutants = ['E' if m.is_equivalent else ' ' for m in mutants]
    marked_killed_mutants = ['K' if m.is_killed else ' ' for m in mutants]
    for i, mutant in enumerate(mutants):
        score_line = []
        mutant_package = 'mujava.program_session.result.Algorithm.traditional_mutants.' \
                         + mutant.method + '.' + mutant.operator + '.Algorithm'
        Algorithm = jpype.JClass(mutant_package)
        alg = Algorithm()
        for j, test_case in enumerate(test_set):
            output = run_java_method(alg, method_name, test_case)
            is_correct = test_case.output == output
            score_line.append(is_correct)
            if not is_correct:
                mutant.is_killed = True
        inverted_mutation_table.append(score_line)
    mutation_table = list(map(list, zip(*inverted_mutation_table)))
    for i, test_case in enumerate(test_set):
        test_case.mutant_proportion = mutation_table[i].count(False) / len(mutation_table[i])

    count_score = [all(col) for col in inverted_mutation_table].count(False)
    if print_table:
        print_mutation_table(mutation_table, test_set, marked_eq_mutants, marked_killed_mutants)
        print('all score', count_score, '/', len(mutants))
        print('non eq score', count_score, '/', considered_non_eq_mutants_length)
    score = count_score / len(mutants)
    non_eq_score = count_score / considered_non_eq_mutants_length
    non_eq_mutants_ratio = considered_non_eq_mutants_length / len(mutants)
    return [score, non_eq_score, non_eq_mutants_ratio]


def print_mutation_table(mutation_table, test_set, marked_eq_mutants, marked_killed_mutants):
    print('mutation table')
    print(*marked_eq_mutants)
    print(*marked_killed_mutants)
    for i in range(0, len(mutation_table)):
        print(*bool_array_to_int(mutation_table[i]))
    print('test case info')
    for i in range(0, len(mutation_table)):
        print(test_set[i].input, test_set[i].output, test_set[i].mutant_proportion)


def bool_array_to_int(boolean_list):
    return [1 if x else 0 for x in boolean_list]


def run_java_method(alg, method_name, test_case):
    output = [None]

    def run_alg():
        output[0] = getattr(alg, method_name)(*test_case.input)

    java_thread = threading.Thread(target=run_alg)
    java_thread.daemon = True
    java_thread.start()
    java_thread.join(timeout=1)
    return output[0]


def mark_equivalent_mutants(method_name, mutants):
    print('mark_equivalent_mutants...')
    original_code = get_original_code(method_name)
    code_pairs = []
    for mutant in mutants:
        code_pairs.append([original_code, mutant.method_code])
    results = evaluate(code_pairs, method_name)
    for i in range(0, len(results)):
        mutants[i].set_is_equivalent(results[i])
    return mutants


def get_method_code(code, method_name):
    code_line_list = code.split('\n')
    pos1 = [idx for idx, s in enumerate(code_line_list) if method_name in s][0]
    pos2 = get_method_end_position(code_line_list, pos1)
    assert pos2 != -1
    return '\n'.join(code_line_list[pos1:pos2 + 1])


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


def get_mutants(method_name_filter):
    mutants = []
    for p in Path('./mujava/program_session/result/Algorithm/traditional_mutants').glob('**/*.java'):
        method = pathlib.PurePath(p).parent.parent.name
        method_name = method.split('(')[0].split('_')[1]
        if method_name_filter == method_name:
            operator = pathlib.PurePath(p).parent.name
            method_code = get_method_code(p.read_text(), ' ' + method_name + '(')
            mutants.append(Mutant(p.name, method, operator, method_code))
    return mutants


def get_original_code(method_name):
    source_code = Path('mujava/program_session/result/Algorithm/original/Algorithm.java').read_text()
    return get_method_code(source_code, method_name)


def verify_initial_test_set(method_name, test_set):
    Algorithm = jpype.JClass('mujava.program_session.result.Algorithm.original.Algorithm')
    alg = Algorithm()
    for test_case in test_set:
        output = getattr(alg, method_name)(*test_case.input)
        if test_case.output != output:
            print(test_case.input, output, test_case.output)
            exit()


def main():
    print('main')
    mutants_prep.fix_package_structure()

    os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk1.8.0_351/bin'
    jpype.startJVM(jpype.getDefaultJVMPath())
    apply_mutation_testing_with_test_data_generation('boolean', 'triangle', ['int', 'int', 'int'],
                                                     set_initial_data=True)
    apply_mutation_testing_with_test_data_generation('boolean', 'triangle', ['int', 'int', 'int'])
    jpype.shutdownJVM()
    print('main finished')


if __name__ == "__main__":
    main()
