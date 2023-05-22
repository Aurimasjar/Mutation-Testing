"""
Class for storing data about mutant.
"""

class Mutant:
    def __init__(self, filename, method, operator, method_code):
        self.filename = filename
        self.method = method
        self.operator = operator
        self.code = None
        self.method_code = method_code
        self.is_equivalent = None
        self.is_killed = None
        self.is_valid = True

    def set_method_code(self, method_code):
        self.method_code = method_code

    def set_is_equivalent(self, is_equivalent):
        self.is_equivalent = is_equivalent

    def set_is_valid(self, is_valid):
        self.is_valid = is_valid
