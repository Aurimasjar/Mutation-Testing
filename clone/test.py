from ast import literal_eval

import pandas as pd
import torch
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from ast_model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import plot
import combined_model
from metrics_model import MetricsModel

warnings.filterwarnings('ignore')

"""
Code used to test trained models for both code clone detection and equivalent mutant detection tasks.
Results for models are presented both combined and separately.
Also the graph where combined model accuracy and recall values change by changing metrics model threshold is presented. 
"""

def get_ast_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


def get_metrics_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['metrics_x'])
        x2.append(item['metrics_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    code_x, code_y, metrics_x, metrics_y, labels = [], [], [], [], []
    for _, item in tmp.iterrows():
        code_x.append(item['code_x'])
        code_y.append(item['code_y'])
        metrics_x.append(item['metrics_x'])
        metrics_y.append(item['metrics_y'])
        labels.append([item['label']])
    return code_x, code_y, metrics_x, metrics_y, torch.FloatTensor(labels)

def get_trues_count(arr):
    count = 0
    for x in arr:
        if '1.' in x or True in x or 'True' in x:
            count += 1
    return count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|javamut]")
    parser.add_argument('--lang')
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    categories = 1
    if lang == 'java':
        categories = 5
    print("Test for ", str.upper(lang))
    test_data = pd.read_csv(root + lang + '/test/blocks_and_metrics.csv', converters={
        "code_x": literal_eval, "code_y": literal_eval, "metrics_x": literal_eval, "metrics_y": literal_eval
    }).sample(frac=1)
    METRICS_DIM = 44
    if lang == 'javamut':
        METRICS_DIM = 58
    # if lang == 'java':
    #     for atd_i in range(0, len(test_data['code_x'])-1):
    #         if isinstance(test_data['code_x'][atd_i], float):
    #             test_data['code_x'][atd_i] = [[1]]
    #     for atd_i in range(0, len(test_data['metrics_x']) - 1):
    #         if isinstance(test_data['metrics_x'][atd_i], float):
    #             test_data['metrics_x'][atd_i] = [0] * METRICS_DIM

    word2vec = Word2Vec.load(root + lang + "/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors


    BATCH_SIZE = 32
    USE_GPU = False
    ast_model_filepath = 'output/' + lang + '/ast_model.pkl'
    metrics_model_filepath = 'output/' + lang + '/metrics_model.pkl'
    metadata = pd.read_csv(root + lang + '/train' + '/metadata.csv')
    means, stds = metadata['means'].tolist(), metadata['stds'].tolist()

    ast_model = BatchProgramCC(EMBEDDING_DIM, MAX_TOKENS + 1, BATCH_SIZE, embeddings)
    metrics_model = MetricsModel(METRICS_DIM, BATCH_SIZE, means, stds)
    if USE_GPU:
        ast_model.cuda()

    loss_function = torch.nn.BCELoss()

    ast_precision, ast_recall, ast_f1 = 0, 0, 0
    metrics_precision, metrics_recall, metrics_f1 = 0, 0, 0
    prf_ast_list = []
    prf_metrics_list = []
    for t in range(1, categories + 1):
        if lang == 'java':
            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
            if t != 3:
                continue
        else:
            test_data_t = test_data

        print("Load ast model from ", ast_model_filepath)
        ast_model.load_state_dict(torch.load(ast_model_filepath))
        ast_model.eval()
        print("Load metrics model from ", metrics_model_filepath)
        metrics_model.load_state_dict(torch.load(metrics_model_filepath))
        metrics_model.eval()

        print("Testing ast - %d..." % t)
        # testing procedure
        trues = []
        ast_predicts = []
        ast_predicts_probability = []
        ast_total_loss = 0.0
        ast_total = 0.0
        metrics_predicts = []
        metrics_predicts_probability = []
        metrics_total_loss = 0.0
        metrics_total = 0.0
        i = 0
        while i < len(test_data_t):
            print("test", i, " \ ", len(test_data_t))
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            ast_test1_inputs, ast_test2_inputs, metrics_test1_inputs, metrics_test2_inputs, test_labels = batch
            if USE_GPU:
                ast_test_labels = test_labels.cuda()

            ast_model.batch_size = len(test_labels)
            ast_model.hidden = ast_model.init_hidden()
            ast_output = ast_model(ast_test1_inputs, ast_test2_inputs)
            ast_loss = loss_function(ast_output, Variable(test_labels))

            metrics_model.batch_size = len(test_labels)
            metrics_output = metrics_model(metrics_test1_inputs, metrics_test2_inputs)
            metrics_loss = loss_function(metrics_output, Variable(test_labels))

            # calc testing acc
            ast_predicted = (ast_output.data > 0.5).cpu().numpy()
            ast_predicts.extend(ast_predicted)
            ast_predicts_probability.extend(ast_output.data)
            trues.extend(test_labels.cpu().numpy())
            ast_total += len(test_labels)
            ast_total_loss += ast_loss.item() * len(test_labels)
            metrics_predicted = (metrics_output.data > 0.5).cpu().numpy()
            metrics_predicts.extend(metrics_predicted)
            metrics_predicts_probability.extend(metrics_output.data)
            metrics_total += len(test_labels)
            metrics_total_loss += metrics_loss.item() * len(test_labels)


        plot.plot_confusion_matrix(ast_predicts, trues, 'javamut_ast_confusion_matrix')
        plot.plot_confusion_matrix(metrics_predicts, trues, 'javamut_metrics_confusion_matrix')
        predicts_and = combined_model.combine_and(ast_predicts, metrics_predicts)
        predicts_or = combined_model.combine_or(ast_predicts, metrics_predicts)
        predicts_or_05_015 = combined_model.combine_or_prob(ast_predicts_probability, metrics_predicts_probability, 0.5, 0.15)
        plot.plot_confusion_matrix(predicts_and, trues, 'javamut_combined_and_confusion_matrix')
        plot.plot_confusion_matrix(predicts_or, trues, 'javamut_combined_or_confusion_matrix')
        plot.plot_confusion_matrix(predicts_or_05_015, trues, 'javamut_combined_or_05_015_confusion_matrix')

        ast_cm = confusion_matrix(np.array(ast_predicts), np.array(trues))
        print('ast confusion matrix', ast_cm)
        predicts_or_cms = []
        true_positives = []
        false_positives = []
        false_negatives = []
        true_negatives = []
        thresholds = []
        for thr in np.arange(0, 1, 0.05):
            thresholds.append(thr)
            predicts_or_item = combined_model.combine_or_prob(ast_predicts_probability, metrics_predicts_probability, 0.5, thr)
            cm = confusion_matrix(np.array(predicts_or_item), np.array(trues))
            true_positives.append(cm[1][1])
            false_positives.append(cm[1][0])
            false_negatives.append(cm[0][1])
            true_negatives.append(cm[0][0])
            predicts_or_cms.append(cm)
        print('all confusiom matrices', predicts_or_cms)
        plot.plot_unit_graph_2(thresholds, false_positives, false_negatives, 'Klaidinga tiesa', 'Nerasta tiesa', 'Riba', 'Vienetų kiekis', 'Vienetų skaičius keičiantis metrikų modelio ribai', 'javamut_absolute_unit_graph')
        plot.plot_unit_graph_2(thresholds, [x/ast_cm[0][0] for x in false_positives], [x/ast_cm[0][1] for x in false_negatives], 'Klaidinga tiesa', 'Nerasta tiesa', 'Riba', 'Vienetų dalis', 'Vienetų dalis keičiantis metrikų modelio ribai', 'javamut_relative_unit_graph')

        if lang == 'java':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]

            ast_p, ast_r, ast_f, _ = precision_recall_fscore_support(trues, ast_predicts, average='binary')
            ast_precision += weights[t] * ast_p
            ast_recall += weights[t] * ast_r
            ast_f1 += weights[t] * ast_f
            prf_ast_list.append([ast_p, ast_r, ast_f])
            print("Type-" + str(t) + ": " + str(ast_p) + " " + str(ast_r) + " " + str(ast_f))

            metrics_p, metrics_r, metrics_f, _ = precision_recall_fscore_support(trues, metrics_predicts, average='binary')
            metrics_precision += weights[t] * metrics_p
            metrics_recall += weights[t] * metrics_r
            metrics_f1 += weights[t] * metrics_f
            prf_metrics_list.append([metrics_p, metrics_r, metrics_f])
            print("Type-" + str(t) + ": " + str(metrics_p) + " " + str(metrics_r) + " " + str(metrics_f))
        else:
            ast_precision, ast_recall, ast_f1, _ = precision_recall_fscore_support(trues, ast_predicts, average='binary')
            metrics_precision, metrics_recall, metrics_f1, _ = precision_recall_fscore_support(trues, metrics_predicts, average='binary')
            print('lens of trues, astpr, metricspr', len(trues), len(ast_predicts), len(metrics_predicts))
            print('trues of trues, astpr, metricspr', get_trues_count(trues), get_trues_count(ast_predicts), get_trues_count(metrics_predicts))
            # print('trues, astpr, metricspr', trues, ast_predicts, metrics_predicts)

    print("prf metrics list", prf_metrics_list)
    print("prf ast list", prf_ast_list)
    print("Total ast testing results(P,R,F1):%.3f, %.3f, %.3f" % (ast_precision, ast_recall, ast_f1))
    print("Total metrics testing results(P,R,F1):%.3f, %.3f, %.3f" % (metrics_precision, metrics_recall, metrics_f1))
