import pandas as pd
import torch
import time
import numpy as np
import warnings
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from ast import literal_eval

import model_training
import plot
from metrics_model import MetricsModel

warnings.filterwarnings('ignore')


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['metrics_x'])
        x2.append(item['metrics_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


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
    print("Train for ", str.upper(lang))
    train_data = pd.read_csv(root + lang + '/train/blocks_and_metrics.csv',
        converters={"metrics_x": literal_eval, "metrics_y": literal_eval}).sample(frac=1)
    valid_data = pd.read_csv(root + lang + '/dev/blocks_and_metrics.csv',
        converters={"metrics_x": literal_eval, "metrics_y": literal_eval}).sample(frac=1)
    test_data = pd.read_csv(root + lang + '/test/blocks_and_metrics.csv',
        converters={"metrics_x": literal_eval, "metrics_y": literal_eval}).sample(frac=1)
    METRICS_DIM = 44
    if lang == 'javamut':
        METRICS_DIM = 58
    elif lang == 'java':
        for atd_i in range(0, len(test_data['metrics_x'])-1):
            if isinstance(test_data['metrics_x'][atd_i], float):
                test_data['metrics_x'][atd_i] = [0] * METRICS_DIM

    EPOCHS = 500
    BATCH_SIZE = 32
    USE_GPU = False
    THRESHOLD = 0.5
    model_filepath = 'output/' + lang + '/metrics_model.pkl'

    print('Calculate means and stds...')
    metrics_data = []
    for _, item in train_data.iterrows():
        metrics_data.append(item['metrics_x'])
        metrics_data.append(item['metrics_y'])
    metrics_data = [np.array(list(x)) for x in zip(*metrics_data)]
    means = [np.mean(x) for x in metrics_data]
    stds = [np.std(x) for x in metrics_data]
    metadata = pd.DataFrame({'means': means, 'stds': stds})
    metadata.to_csv((root + lang + '/train' + '/metadata.csv'))

    print('Create model...')
    model = MetricsModel(METRICS_DIM, BATCH_SIZE, means, stds)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    print(train_data)
    train_loss_data, train_acc_data = [], []
    valid_loss_data, valid_acc_data = [], []
    precision, recall, f1 = 0, 0, 0
    prf_list = []
    print("EPOCHS, BATCH SIZE, THRESHOLD", EPOCHS, BATCH_SIZE, THRESHOLD)
    start_time = time.time()
    endd_time = 0
    print('Start training...')
    for t in range(1, categories + 1):
        if lang == 'java':
            train_data_t = train_data[train_data['label'].isin([t, 0])]
            train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

            valid_data_t = valid_data[valid_data['label'].isin([t, 0])]
            valid_data_t.loc[valid_data_t['label'] > 0, 'label'] = 1

            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
            if t != 3:
                continue
        elif lang == 'c':
            train_data_t, valid_data_t, test_data_t = train_data, valid_data, test_data
        else:
            train_data_t, test_data_t = train_data, test_data
        # training procedure
        for epoch in range(EPOCHS):
            print('epoch', epoch)
            epoch_time = time.time()
            print('epoch and start time', epoch, epoch_time)
            # training epoch
            predicts, trues = [], []
            valid_predicts, valid_trues = [], []
            total_loss = 0.0
            total = 0.0
            valid_total_loss = 0.0
            valid_total = 0.0
            i = 0
            model.train()
            while i < len(train_data_t):
                # print('train', i, ' / ', len(train_data_t))
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                output = model(train1_inputs, train2_inputs)
                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

                predicts.extend((output.data > THRESHOLD).cpu().numpy())
                trues.extend(train_labels.cpu().numpy())
                total += len(train_labels)
                total_loss += loss.item() * len(train_labels)

            train_acc = model_training.count_accuracy(trues, predicts)

            train_acc_data.append(train_acc)
            train_loss_data.append(total_loss / total)

            if lang != 'javamut':
                i = 0
                model.eval()
                while i < len(valid_data_t):
                    # print("validation", i, " \ ", len(valid_data_t))
                    batch = get_batch(valid_data_t, i, BATCH_SIZE)
                    i += BATCH_SIZE
                    valid1_inputs, valid2_inputs, valid_labels = batch
                    if USE_GPU:
                        valid1_inputs, valid2_inputs, valid_labels = valid1_inputs, valid2_inputs, valid_labels.cuda()

                    output = model(valid1_inputs, valid2_inputs)
                    loss = loss_function(output, Variable(valid_labels))
                    valid_total_loss += loss.item() * len(valid_labels)
                    valid_predicts.extend((output.data > THRESHOLD).cpu().numpy())
                    valid_trues.extend(valid_labels.cpu().numpy())
                    valid_total += len(valid_labels)

                valid_acc = model_training.count_accuracy(valid_trues, valid_predicts)

                valid_acc_data.append(valid_acc)
                valid_loss_data.append(valid_total_loss / valid_total)

        endd_time = time.time()
        print("Training finished time", endd_time)
        print("Saving model to ", model_filepath)
        torch.save(model.state_dict(), model_filepath)
        print('train_loss_data', train_loss_data)
        print('train_acc_data', train_acc_data)

        if lang != 'javamut':
            print('valid_loss_data', valid_loss_data)
            print('valid_acc_data', valid_acc_data)
            plot.plot_training_loss_stats(train_loss_data, valid_loss_data, 'java_metrics_loss_function')
            plot.plot_training_acc_stats(train_acc_data, valid_acc_data, 'java_metrics_acc_function')
        else:
            plot.plot_training_loss_wv_stats(train_loss_data, 'javamut_metrics_loss_function')
            plot.plot_training_acc_wv_stats(train_acc_data, 'javamut_metrics_acc_function')

        print("Testing-%d..." % t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            # print("test", i, " \ ", len(test_data_t))
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            output = model(test1_inputs, test2_inputs)
            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > THRESHOLD).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)

        plot.plot_confusion_matrix(predicts, trues, 'javamut_metrics_confusion_matrix')
        if lang == 'java':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
            prf_list.append([p, r, f])
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            prf_list.append([precision, recall, f1])

    print("prf list", prf_list)
    print("Total testing results(P,R,F1):%.3f & %.3f & %.3f" % (precision, recall, f1))
    end_time = time.time()
    print("End time", end_time)
    diff = endd_time - start_time
    print("Experiment lasted ", diff, " seconds (", diff/60, "minutes)")
