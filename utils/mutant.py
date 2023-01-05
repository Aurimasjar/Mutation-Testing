class Mutant:
    def __init__(self, filename, method, operator, code):
        self.filename = filename
        self.method = method
        self.operator = operator
        self.code = code
        self.parsed_code = None
        self.vector = None
        self.is_equivalent = None

    def set_is_equivalent(self, is_equivalent):
        self.is_equivalent = is_equivalent
