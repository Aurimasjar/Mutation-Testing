import jpype

"""
Class for storing data about test case.
Field bit_input represents test case converted to stream of bits.
Current version supports only static number of fields without arrays.
"""


class TestCase:
    def __init__(self, test_case, bit_size, bit_input=None, method_name=None):
        self.bit_size = bit_size
        self.mutant_proportion = None
        if bit_input is not None:
            self.bit_input = bit_input
            self.convert_from_bits()
            assert method_name is not None
            self.recalculate_output(method_name)
        else:
            self.input = test_case[:-1]
            self.output = test_case[-1]
            self.element_sizes = [len(x) if isinstance(x, list) else 1 for x in test_case]
            self.all_element_count = sum(self.element_sizes)
            self.bit_input = None

    def convert_to_bits(self):
        param_bits = []
        for param in self.input:
            param_bits.append("{0:b}".format(param).zfill(self.bit_size))
        self.bit_input = ''.join(param_bits)

    def convert_from_bits(self):
        param_bits = [self.bit_input[i:i + self.bit_size] for i in range(0, len(self.bit_input), self.bit_size)]
        self.input = [int(param, 2) for param in param_bits]

    def recalculate_output(self, method_name):
        Algorithm = jpype.JClass('mujava.program_session.result.Algorithm.original.Algorithm')
        alg = Algorithm()
        self.output = getattr(alg, method_name)(*self.input)

    def print_test_case(self):
        print('input and output', self.input, self.output)

    def print_bit_input(self):
        print('bit input size', len(self.bit_input))
        print('bit size', self.bit_size)
        print('bit input', self.bit_input)
