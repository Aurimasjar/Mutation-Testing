import os
from pathlib import Path
import jpype
from jpype import java

import pathlib

import javalang
import pandas as pd

from clone.eval import evaluate
from constants import absolute_path
from genetic_algorithm import mutate, crossover
from mutant import Mutant
from test_case import TestCase


def print_mutant_list(mutants):
    print('length of mutant list', len(mutants))
    # for i in range(0, len(mutants)):
    #     print('mutant', i, mutants[i].filename, mutants[i].method, mutants[i].operator)


def apply_mutation_testing(return_type, method_name, method_params):
    print('power_mutation_testing')
    test_set_data = pd.read_csv(
        'programs/' + form_method_signature(return_type, method_name, method_params) + 'test_set.csv')
    print('test set data', test_set_data)
    test_set = []
    for test_case in test_set_data.values:
        test_set.append(TestCase(test_case, 8))

    mutants = get_mutants()
    mutants = list(
        filter(lambda m: m.method == form_method_signature(return_type, method_name, method_params), mutants))
    print_mutant_list(mutants)
    verify_initial_test_set(method_name, test_set)
    mutants = kill_equivalent_mutants(method_name, mutants)
    print_mutant_list(mutants)

    scores = []
    test_set = convert_to_bits(test_set)
    for i in range(0, 3):
        print('genetic algorithm iteration', i)
        score = evaluate_mutation_score(method_name, mutants, test_set)
        scores.append(score)
        apply_genetic_operations(test_set)
        test_set = convert_from_bits(test_set)
        recalculate_outputs(test_set, method_name)
    score = evaluate_mutation_score(method_name, mutants, test_set)
    scores.append(score)
    print('scores', scores)


def convert_to_bits(test_set):
    for test_case in test_set:
        test_case.convert_to_bits()
    return test_set


def convert_from_bits(test_set):
    for test_case in test_set:
        test_case.convert_to_bits()
    return test_set

def recalculate_outputs(test_set, method_name):
    for test_case in test_set:
        test_case.recalculate_output(method_name)
    return test_set


def apply_genetic_operations(test_set):
    # print('apply_genetic_operations')
    removal_threshold = 0.2
    threshold = 0.5
    amount_of_test_cases = len(test_set)

    # todo implement genetic algorithm
    # # selection
    # for test_case in test_set:
    #     if test_case.mutant_proportion <= removal_threshold:
    #         test_set.remove(test_case)
    #
    # # mutation
    # for test_case in test_set:
    #     if test_case.mutant_proportion <= threshold:
    #         mutate(test_case.bit_input)
    #
    # # crossover
    # for test_case in test_set:
    #     print('generate new test cases')
    # while len(test_set) < amount_of_test_cases:
    #     test_set.append(crossover(test_set))


def form_method_signature(return_type, method_name, method_params):
    return return_type + '_' + method_name + '_' + '_'.join(method_params) + '_'


def evaluate_mutation_score(method_name, mutants, test_set):
    print('evaluate_mutation_score')
    print('length of mutant list', len(mutants))
    score = 0
    for mutant in mutants:
        print(mutant.operator)
        mutant_package = 'mujava.result.Algorithm.traditional_mutants.' + mutant.method + '.' + mutant.operator + '.Algorithm'
        Algorithm = jpype.JClass(mutant_package)
        alg = Algorithm()
        for test_case in test_set:
            print(test_case.input, test_case.output)
            # todo solve infinite loop problem
            output = getattr(alg, method_name)(*test_case.input)
            print('output', output)
            # print('mutant and test_case', mutant.operator, test_case.input, output)
            if test_case.output != output:
                mutant.is_killed = True
                score += 1
    print('score', score, '/', len(mutants))
    return score / len(mutants)


def kill_equivalent_mutants(method_name, mutants):
    print('kill_equivalent_mutants')
    # code1 = Path('mujava/VendingMachine.java').read_text()
    # code2 = Path('mujava/VendingMachineM.java').read_text()
    original_code = get_original_code()
    print('original code clone detection results')
    evaluate([[original_code, original_code]], method_name)

    print('mutant detection results')
    code_pairs = []
    for mutant in mutants:
        code_pairs.append([original_code, mutant.code])
    # print('code pairs', code_pairs)
    results = evaluate(code_pairs, method_name)
    print('results', results)
    for i in range(0, len(results)):
        mutants[i].set_is_equivalent(results[i])
        # print('mutant', i, mutants[i].__dict__)
    return list(filter(lambda m: m.is_equivalent is False, mutants))


def get_mutants():
    mutants = []
    for p in Path('./mujava/result/Algorithm/traditional_mutants').glob('**/*.java'):
        method = pathlib.PurePath(p).parent.parent.name
        operator = pathlib.PurePath(p).parent.name
        mutants.append(Mutant(p.name, method, operator, p.read_text()))
    return mutants


def get_original_code():
    return Path('mujava/result/Algorithm/original/Algorithm.java').read_text()


def verify_initial_test_set(method_name, test_set):
    print('verify_initial_test_set')
    Algorithm = jpype.JClass('mujava.result.Algorithm.original.Algorithm')
    alg = Algorithm()
    for test_case in test_set:
        output = getattr(alg, method_name)(*test_case.input)
        if test_case.output != output:
            print(test_case.input, output, test_case.output)
            exit()


def main():
    print('main')
    os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk1.8.0_351/bin'
    jpype.startJVM(jpype.getDefaultJVMPath())
    apply_mutation_testing('int', 'power', ['int', 'int'])
    jpype.shutdownJVM()
    print('main finished')


if __name__ == "__main__":
    main()
