import time
from ast import literal_eval

import numpy as np
import pandas as pd
import os
import sys
import warnings
import click
from tqdm.auto import tqdm

import metrics

tqdm.pandas()
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, ratio, root, language: str):

        self.language = language.lower()
        assert self.language in ('c', 'java', 'javamut')
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.metrics = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def get_parsed_source(self, input_file: str,
                          output_file: str = None) -> pd.DataFrame:
        """Parse code using pycparser

        If the user doesn't provide `output_file`, the method reads the
        a DataFrame containing the columns id, code (C/Java code parsed
        by pycparser) and label. Otherwise it reads a Dataframe from
        `input_file` containing the columns id, code (input C/Java code)
        and label, applies the c_parser/javalang to the code column and
        stores the resulting dataframe into `output_file`

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file

        Returns:
            pd.DataFrame: DataFrame with the columns id, code (C/Java code
                parsed by pycparser/javalang) and label.
        """
        input_path = os.path.join(self.root, self.language, input_file)
        if output_file is None:
            source = pd.read_pickle(input_path)
        else:
            output_path = os.path.join(self.root, self.language, output_file)
            if self.language == 'c':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_csv(input_path)
                source.columns = ['id', 'code', 'label']
                source['code'] = source['code'].progress_apply(parser.parse)
                source.to_csv('ast.csv')
                source.to_pickle(output_path)
            else:
                import javalang

                def parse_program(func):
                    tokens = javalang.tokenizer.tokenize(func, True)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree

                if self.language == 'java':
                    source = pd.read_csv(input_path, delimiter='\t')
                else:
                    source = pd.read_csv(input_path)
                source.columns = ['id', 'code']
                source['source_code'] = source['code']
                source['code'] = source['code'].progress_apply(parse_program)
                source.to_csv('ast.csv')
                source.to_pickle(output_path)
        self.sources = source
        return source

    def read_pairs(self, filename: str):
        """Create clone pairs

        Args:
            filename (str): [description]
        """
        if self.language == 'javamut':
            self.pairs = pd.read_csv(os.path.join(self.root, self.language,
                                            filename))
        else:
            self.pairs = pd.read_csv(os.path.join(self.root, self.language,
                                            filename))
            if self.language == 'java':
                # fix pairs list by removing pairs with non-existent code ids from the original bcb dataset
                filtered_pairs = self.pairs[~self.pairs['id1'].isin(
                    [1032896, 74, 2524323, 5180407, 8643644, 15503077, 12639648, 19727309, 20395377, 8040734, 22237273]
                )]

    # calculate metrics for each ast
    def calculate_metrics(self, output_file):
        if self.language == 'c':
            self.metrics = pd.DataFrame(self.sources['code'].progress_apply(metrics.calculate_c_metrics))
        elif self.language == 'java':
            self.metrics = pd.DataFrame(self.sources['code'].progress_apply(metrics.calculate_java_metrics))
        else:
            self.metrics = pd.DataFrame(self.sources['code'].progress_apply(metrics.calculate_java_mut_metrics))


        print('self.metrics', self.metrics)
        # write dataset metrics metadata to metrics_data csv file
        metrics_data = [np.array(list(x)) for x in zip(*self.metrics['code'])]
        means = [str(round(np.mean(x), 3)) for x in metrics_data]
        stds = [str(round(np.std(x), 3)) for x in metrics_data]
        metadata = pd.DataFrame({'means': means, 'stds': stds})
        metadata.to_csv((self.root + self.language + '/metrics_data_rounded.csv'))

        self.sources['metrics'] = self.metrics
        output_path = os.path.join(self.root, self.language, output_file)
        self.sources['metrics'].to_csv(output_path)
        self.metrics = self.sources

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root + self.language + '/'
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.csv'
        train.to_csv(self.train_file_path)

        dev_path = data_path + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.csv'
        dev.to_csv(self.dev_file_path)

        test_path = data_path + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.csv'
        test.to_csv(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root + self.language + '/'
        if not input_file:
            input_file = self.train_file_path
        pairs = pd.read_csv(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id', drop=False).loc[train_ids]
        if not os.path.exists(data_path + 'train/embedding'):
            os.mkdir(data_path + 'train/embedding')
        if self.language == 'c':
            sys.path.append('../')
            from prepare_data_c import get_sequences as func
        else:
            from prepare_data_java import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            # print('seq', sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(data_path + 'train/programs_ns.csv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, vector_size=size, workers=16, sg=1,
                       max_final_vocab=3000)
        w2v.save(data_path + 'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language == 'c':
            from prepare_data_c import get_blocks as func
        else:
            from prepare_data_java import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(
            self.root + self.language + '/train/embedding/node_w2v_' +
            str(self.size)
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
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.DataFrame(self.sources, copy=True)
        trees['code'] = trees['code'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self, data_path, part):
        pairs = pd.read_csv(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left',
                      left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left',
                      left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)

        df.to_csv(self.root + self.language + '/' + part + '/blocks_and_metrics.csv')


    # run for processing data to train
    def run(self):
        print('parse source code...')
        input_file = ''
        if self.language == 'c':
            input_file = 'programs.csv'
        elif self.language == 'java':
            input_file = 'bcb_funcs_all.csv'
        else:
            input_file = 'mut_funcs_all.csv'
        if os.path.exists(os.path.join(self.root, self.language, 'ast.pkl')):
            print('a')
            self.get_parsed_source(input_file='ast.pkl')
        else:
            self.get_parsed_source(input_file=input_file,
                                   output_file='ast.pkl')
        print('read id pairs...')
        if self.language == 'c':
            self.read_pairs('oj_clone_ids.csv')
        elif self.language == 'java':
            self.read_pairs('bcb_pair_ids.csv')
        else:
            self.read_pairs('mut_pair_ids.csv')

        print('calculate metrics...')
        self.calculate_metrics('metrics.csv')

        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None, 128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')


    def test(self):
        source = pd.read_csv(os.path.join(self.root, self.language, 'mut_funcs_all.csv'))
        pairs = pd.read_csv(os.path.join(self.root, self.language, 'mut_pair_ids.csv'))
        print('source and pairs first element', source[0], pairs[0])


@click.command()
@click.option('--lang', required=True, type=str,
              help="Language for the code input ('c', 'java or javamut')")
def main(lang):
    split = '4:0:1' if lang == 'javamut' else '3:1:1'
    ppl = Pipeline(split, 'data/', str(lang))
    ppl.run()
    # ppl.test()
    print("finished")


if __name__ == "__main__":
    main()
