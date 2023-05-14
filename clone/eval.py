from pathlib import Path

import javalang
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from javalang.tree import ClassDeclaration, MethodDeclaration

from .ast_model import BatchProgramCC
from .combined_model import combine_or_prob
from .metrics import calculate_java_mut_metrics
from .metrics_model import MetricsModel
from .prepare_data_java import get_sequence, get_blocks_v1
from .tree import SingleNode

root = 'clone/data/'
lang = 'javamut'
ast_model_filepath = 'clone/output/' + lang + '/ast_model.pkl'
metrics_model_filepath = 'clone/output/' + lang + '/metrics_model.pkl'


def evaluate(code_pairs, method_name):
    BATCH_SIZE = len(code_pairs)
    EMBEDDING_DIM, MAX_TOKENS, embeddings = load_embeddings()
    metadata = pd.read_csv(root + lang + '/train' + '/metadata.csv')
    METRICS_DIM = 58
    means, stds = metadata['means'].tolist(), metadata['stds'].tolist()
    ast_model = BatchProgramCC(EMBEDDING_DIM, MAX_TOKENS + 1, BATCH_SIZE, embeddings)
    metrics_model = MetricsModel(METRICS_DIM, BATCH_SIZE, means, stds)
    # print("Load ast model from ", ast_model_filepath)
    ast_model.load_state_dict(torch.load(ast_model_filepath))
    ast_model.eval()
    # print("Load metrics model from ", metrics_model_filepath)
    metrics_model.load_state_dict(torch.load(metrics_model_filepath))
    metrics_model.eval()

    ast1 = []
    ast2 = []
    metrics1 = []
    metrics2 = []
    for pair in code_pairs:
        ast11, ast22, metrics11, metrics22 = construct_code_info(pair, method_name)
        ast1.append(ast11)
        ast2.append(ast22)
        metrics1.append(metrics11)
        metrics2.append(metrics22)

    ast_predicts_probability = []
    metrics_predicts_probability = []

    ast_model.batch_size = len(ast1)
    ast_output = ast_model(ast1, ast2)
    metrics_model.batch_size = len(metrics1)
    metrics_output = metrics_model(metrics1, metrics2)
    # for i in range (0, len(ast_output.data.cpu().numpy())):
    #     print('[i, ast_prob, metrics_prob]', i, ast_output.data.cpu().numpy()[i][0], metrics_output.data.cpu().numpy()[i][0])
    ast_predicts_probability.extend(ast_output.data)
    metrics_predicts_probability.extend(metrics_output.data)
    result = combine_or_prob(ast_predicts_probability, metrics_predicts_probability, 0.1, 0.2)
    return list(map(lambda x: x[0], result))


def construct_code_info(pair, method_name):
    # print('construct_code_info', pair[0], pair[1])
    # source['code'] = parse_program(pair[0])
    ast1 = parse_program(pair[0])
    ast2 = parse_program(pair[1])

    ast1 = get_method_ast(ast1, method_name)
    ast2 = get_method_ast(ast2, method_name)

    metrics1 = calculate_java_mut_metrics(ast1)
    metrics2 = calculate_java_mut_metrics(ast2)

    # split asts to blocks like in original paper
    ast1 = generate_block_seqs(ast1)
    ast2 = generate_block_seqs(ast2)

    # print('ast1', ast1)
    # print('ast2', ast2)
    # print('metrics1', metrics1)
    # print('metrics2', metrics2)
    return ast1, ast2, metrics1, metrics2


def get_method_ast(ast, method_name):
    # print('get_method_ast', ast, method_name)
    for path, node in ast.filter(MethodDeclaration):
        if node.name == method_name:
            return node

# generate block sequences with index representations
def generate_block_seqs(ast):
    from gensim.models.word2vec import Word2Vec
    word2vec = Word2Vec.load(
        root + lang + '/train/embedding/node_w2v_' +
        str(128)
    ).wv
    vocab = word2vec.key_to_index
    max_token = word2vec.vectors.shape[0]

    def tree_to_index(node):
        token = node.token
        result = [vocab[token] if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        get_blocks_v1(r, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree

    return trans2seq(ast)


def parse_program(source_code):
    tokens = javalang.tokenizer.tokenize(source_code, True)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    # tree = javalang.parse.parse(source_code)
    return tree


def load_embeddings():
    word2vec = Word2Vec.load(root + lang + "/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
    return EMBEDDING_DIM, MAX_TOKENS, embeddings

# print('xxx')
# code1 = Path('../mujava/VendingMachine.java').read_text()
# code2 = Path('../mujava/VendingMachineM.java').read_text()
# code_pairs = [[code1, code2]]
# evaluate(code_pairs)
