import random


# todo implement genetic algorithm operations

def mutate(bit_input):
    pos = random.randrange(0, len(bit_input))
    bit_input = bit_input[:pos] + invert_bit(bit_input[pos]) + bit_input[pos+1:]
    return bit_input


def invert_bit(bit):
    if bit == '1':
        return '0'
    else:
        return '1'


def crossover(test_set):
    # choose test cases for crossover
    pos1, pos2 = generate_random_numbers(len(test_set))
    assert len(test_set[pos1].bit_input) == len(test_set[pos2].bit_input)

    # initialize crossover points
    test_case_bit_size = len(test_set[0].bit_input)
    cp1, cp2 = generate_random_numbers(test_case_bit_size)

    new_bit_input = test_set[pos1].bit_input[:cp1] + test_set[pos2].bit_input[cp1:cp2] + test_set[pos1].bit_input[cp2:]
    return new_bit_input

def generate_random_numbers(range):
    pos1 = random.randrange(0, range)
    pos2 = pos1
    while pos1 == pos2:
        pos2 = random.randrange(0, range)
    if pos1 < pos2:
        return pos1, pos2
    return pos2, pos1

def generate_bit_input(bit_size):
    input = random.randrange(0, pow(2, bit_size))
    return "{0:b}".format(input).zfill(bit_size)