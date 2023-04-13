import jpype

from constants import absolute_path


class TestCase:
    def __init__(self, test_case, bit_size):
        self.input = test_case[:-1]
        self.output = test_case[-1]
        self.bit_input = None
        self.bit_size = bit_size
        self.mutant_proportion = None

    def convert_to_bits(self):
        # print('convert_to_bits')
        param_bits = []
        for param in self.input:
            param_bits.append("{0:b}".format(param).zfill(self.bit_size))
        self.bit_input = ''.join(param_bits)

    def convert_from_bits(self):
        # print('convert_from_bits')
        param_bits = [self.bit_input[i:i + self.bit_size] for i in range(0, len(self.bit_input), self.bit_size)]
        self.input = [int(param, 2) for param in param_bits]

    def recalculate_output(self, method_name):
        # print('recalculate_output')
        Algorithm = jpype.JClass('mujava.result.Algorithm.original.Algorithm')
        alg = Algorithm()
        self.output = getattr(alg, method_name)(*self.input)

    def print_test_case(self):
        print('input and output', self.input, self.output)

    def print_bit_input(self):
        print('bit input size', len(self.bit_input))
        print('bit size', self.bit_size)
        print('bit input', self.bit_input)
