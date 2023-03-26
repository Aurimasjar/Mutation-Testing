import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def save_to_csv(obj, filename):
    df = pd.DataFrame(obj)
    df.to_csv('images/data/' + filename + '.csv', index=False)


# object must be the same structure as it was saved with save_to_csv
def read_from_csv(filename):
    return pd.read_csv('images/data/' + filename + '.csv')


def plot_training_stats(train_loss, train_acc, filename='plot_training_stats'):
    save_to_csv({'v1': train_loss, 'v2': train_acc}, filename)
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label='Modelio mokymo praradimo funkcija')
    plt.plot(range(epochs), train_acc, label='Modelio mokymo tikslumas')
    plt.xlabel('Epocha')
    plt.ylabel('Modelio mokymo parametras')
    plt.title('Modelio mokymo duomenys')
    plt.legend()
    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_training_loss_stats(train_vector, valid_vector, filename='plot_training_loss_stats'):
    save_to_csv({'v1': train_vector, 'v2': valid_vector}, filename)
    epochs = len(train_vector)
    plt.plot(range(epochs), train_vector, label='Modelio mokymo praradimo funkcija')
    plt.plot(range(epochs), valid_vector, label='Modelio validavimo praradimo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Praradimo funkcija')
    plt.title('Modelio mokymo praradimo funkcijos raida')
    plt.legend()
    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_training_acc_stats(train_vector, valid_vector, filename='plot_training_acc_stats'):
    save_to_csv({'v1': train_vector, 'v2': valid_vector}, filename)
    epochs = len(train_vector)
    plt.plot(range(epochs), train_vector, label='Modelio mokymo tikslumo funkcija')
    plt.plot(range(epochs), valid_vector, label='Modelio validavimo tikslumo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title('Modelio mokymo tikslumas')
    plt.legend()
    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_read_stats(filename='plot_training_acc_stats'):
    vector1 = read_from_csv(filename)['v1']
    vector2 = read_from_csv(filename)['v2']
    epochs = len(vector1)
    plt.plot(range(epochs), vector1, label='Modelio pavyzdinė funkcija')
    plt.plot(range(epochs), vector2, label='Modelio pavyzdinė funkcija 2')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title('Modelio pavyzdinis grafikas')
    plt.legend()
    plt.show()


def plot_unit_graph(xdata, vector, label, xlabel, ylabel, title, filename='plot_unit_graph'):
    save_to_csv({'x': xdata, 'v1': vector}, filename)
    plt.plot(xdata, vector, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_unit_graph_2(xdata, vector1, vector2, label1, label2, xlabel, ylabel, title, filename='plot_unit_graph'):
    save_to_csv({'x': xdata, 'v1': vector1, 'v2': vector2}, filename)
    plt.plot(xdata, vector1, label=label1)
    plt.plot(xdata, vector2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_confusion_matrix(predicts, trues, filename='plot_confusion_matrix'):
    predicts = np.array([p[0] for p in predicts])
    trues = np.array([t[0] for t in trues])
    save_to_csv({'predicts': predicts, 'trues': trues}, filename)
    cm = confusion_matrix(predicts, trues)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neigiamas', 'Teigiamas'], rotation=45)
    plt.yticks(tick_marks, ['Neigiamas', 'Teigiamas'])

    # annotate the confusion matrix with the values
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Tiesos žymė')
    plt.ylabel('Spėjimas')
    plt.subplots_adjust(bottom=0.2)  # adjust the bottom padding

    plt.savefig('images/' + filename + '.png')
    plt.show()


def plot_read_confusion_matrix(filename='plot_confusion_matrix'):
    # filename = 'c_metrics_tv_overfitted_confusion_matrix'
    predicts = np.array(read_from_csv(filename)['predicts'])
    trues = np.array(read_from_csv(filename)['trues'])
    cm = confusion_matrix(predicts, trues)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neigiamas', 'Teigiamas'], rotation=45)
    plt.yticks(tick_marks, ['Neigiamas', 'Teigiamas'])

    # annotate the confusion matrix with the values
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Tiesos žymė')
    plt.ylabel('Spėjimas')
    plt.subplots_adjust(bottom=0.2)  # adjust the bottom padding
    plt.show()
