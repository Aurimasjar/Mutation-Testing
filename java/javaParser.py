import javalang


def testing_javalang():
    print('JavaTest.java parser')
    tree = javalang.parse.parse("package javaMutants; class JavaTest {}")
    print('package name is', tree.package.name)
    print('type is', str(tree.types[0]))
    print('class name is', tree.types[0].name)
    print('tree structure:')
    for path, node in tree:
        print('path:', str(path), ' node:', str(node))
    print('=====')
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        print('path:', str(path), ' node:', str(node))

    print()
    testTokens = list(javalang.tokenizer.tokenize('System.out.println("Hello " + "world");'))
    print('tokens', testTokens)
    print('token values', list(map(lambda token: token.value, testTokens)))
    tokens = javalang.tokenizer.tokenize('System.out.println("Hello " + "world");')
    parser = javalang.parser.Parser(tokens)
    print(parser.parse_expression())

    sourceFilename = "java/source/JavaTest.java"
    parsedFilename = "java/parsed/parsedJavaTest.txt"
    fp = open(sourceFilename, 'r')
    JavaCodeStr = fp.read()
    fp.close()
    tree = javalang.parse.parse(JavaCodeStr)
    tokens = list(javalang.tokenizer.tokenize(JavaCodeStr))
    for t in tokens:
        print(t)

    fw = open(parsedFilename, 'w')
    for t in tokens:
        fw.write(str(t) + '\n')
    fw.close()
    print('javaParser done')


def parse_mutant(mutant):
    mutant.parsed_code = list(javalang.tokenizer.tokenize(mutant.code))


def parse_mutants(mutants):
    for mutant in mutants:
        parse_mutant(mutant)
