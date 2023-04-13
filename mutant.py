class Mutant:
    def __init__(self, filename, method, operator, code):
        self.filename = filename
        self.method = method
        self.operator = operator
        self.code = code
        self.method_code = None
        self.is_equivalent = None
        self.is_killed = None

    def set_method_code(self, method_code):
        self.method_code = method_code

    def set_is_equivalent(self, is_equivalent):
        self.is_equivalent = is_equivalent
