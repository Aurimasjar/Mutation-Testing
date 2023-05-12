import os
import re
import subprocess

initial_mujava_path = 'mujava/result/Algorithm/'
absolute_path = '/'


def fix_package_structure():
    # set package names for original code
    original_code_path = initial_mujava_path + 'original/Algorithm.java'
    set_package(original_code_path)  # or rewrite package if source codes includes package keyword
    compile_java(original_code_path)

    # rename directories to have valid package names
    method_list = [d for d in next(os.walk(initial_mujava_path + 'traditional_mutants'))[1]]
    for method in method_list:
        updated_method = re.sub('[(),]', '_', method)
        source = absolute_path + '/' + initial_mujava_path + 'traditional_mutants/' + method
        destination = absolute_path + '/' + initial_mujava_path + 'traditional_mutants/' + updated_method
        os.rename(source, destination)

    # set package names for mutants
    method_list = [d for d in next(os.walk(initial_mujava_path + 'traditional_mutants'))[1]]
    for method in method_list:
        operator_list = [d for d in next(os.walk(initial_mujava_path + 'traditional_mutants/' + method))[1]]
        for operator in operator_list:
            mutant_path = initial_mujava_path + 'traditional_mutants/' + method + '/' + operator + '/Algorithm.java'
            set_package(mutant_path)   # or rewrite package if source codes includes package keyword
            compile_java(mutant_path)


def set_package(code_path):
    package_name = '.'.join(code_path.split('/')[:-1])
    with open(code_path, 'r', encoding='utf-8') as file:
        code = file.readlines()
    code.insert(0, 'package ' + package_name + ';\n\n')
    with open(code_path, 'w', encoding='utf-8') as file:
        file.writelines(code)


def rewrite_package(code_path):
    package_name = '.'.join(code_path.split('/')[:-1])
    with open(code_path, 'r', encoding='utf-8') as file:
        code = file.readlines()
    index = [idx for idx, s in enumerate(code) if 'package' in s][0]
    code[index] = 'package ' + package_name + ';\n'
    with open(code_path, 'w', encoding='utf-8') as file:
        file.writelines(code)


def compile_java(java_file):
    jdk_path = 'C:/Program Files/Java/jdk1.8.0_351/bin'
    env = os.environ.copy()
    env['PATH'] = jdk_path + ';' + env['PATH']

    cmd = 'javac ' + java_file
    proc = subprocess.Popen(cmd, shell=True, env=env)


fix_package_structure()
