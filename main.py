import re
from pathlib import Path
import numpy as np
import pathlib

from utils.mutant import Mutant
from java import javaParser
from nlp import word2vec
import train


def get_mutants():
    mutants = []
    for p in Path('./mujava/result/VendingMachine/traditional_mutants').glob('**/*.java'):
        method = pathlib.PurePath(p).parent.parent.name
        operator = pathlib.PurePath(p).parent.name
        mutants.append(Mutant(p.name, method, operator, p.read_text()))
    return mutants


# classify mutants using data from file, generated with mujava tool
def get_mutant_classification_map():
    mutant_map = {}
    fp = open('./mujava/output.txt', 'r')
    test_report = [line.rstrip() for line in fp]
    fp.close()
    index = test_report.index('======================================== Generating Original Test Results '
                              '========================================')
    test_case_count = test_report[index + 1].count('pass')
    for line in test_report:
        if line.startswith('  '):
            operator = re.split('[ {]', line)[2]
            is_equivalent = line.count('pass') == test_case_count
            mutant_map[operator] = is_equivalent

    return mutant_map


def classify_mutants(mutants):
    mutant_map = get_mutant_classification_map()
    for mutant in mutants:
        mutant.set_is_equivalent(mutant_map[mutant.operator]),
        vector_array = np.array(mutant.vector)
        if mutant.is_equivalent:
            np.savetxt('./data/equivalent/' + mutant.operator + '.csv', vector_array, delimiter=',')
        else:
            np.savetxt('./data/non_equivalent/' + mutant.operator + '.csv', vector_array, delimiter=',')
        # print(mutant.__dict__)


def prepare_dataset():
    mutants = get_mutants()
    print('mutants collected')
    javaParser.parse_mutants(mutants)
    print('mutant parsing finished')
    word2vec.get_vectors(mutants)
    print('worc2vec finished')
    classify_mutants(mutants)
    print('mutants classified')


def main():
    print('main')
    prepare_dataset()
    train.train_model(32, 10)
    print('main finished')


if __name__ == "__main__":
    main()
