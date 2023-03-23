import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_training_stats(train_loss, train_acc):
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label='Modelio mokymo praradimo funkcija')
    plt.plot(range(epochs), train_acc, label='Modelio mokymo tikslumas')
    plt.xlabel('Epocha')
    plt.ylabel('Modelio mokymo parametras')
    plt.title('Modelio mokymo duomenys')
    plt.legend()
    plt.show()


def plot_training_loss_stats(vector):
    epochs = len(vector)
    plt.plot(range(epochs), vector, label='Modelio mokymo praradimo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Praradimo funkcija')
    plt.title('Modelio mokymo praradimo funkcijos raida')
    plt.legend()
    plt.show()


def plot_training_acc_stats(vector):
    epochs = len(vector)
    plt.plot(range(epochs), vector, label='Modelio mokymo tikslumo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title('Modelio mokymo tikslumas')
    plt.legend()
    plt.show()


def plot_acc_stats(vector, loc='images/acc_image.png'):
    epochs = len(vector)
    plt.plot(range(epochs), vector, label='Modelio mokymo tikslumo funkcija')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title('Modelio mokymo tikslumas')
    plt.legend()
    plt.savefig(loc)
    plt.show()

def plot_unit_graph(xdata, vector, label, xlabel, ylabel, title):
    plt.plot(xdata, vector, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_unit_graph_2(xdata, vector1, vector2, label1, label2, xlabel, ylabel, title):
    plt.plot(xdata, vector1, label=label1)
    plt.plot(xdata, vector2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_confusion_matrix(predicts, trues):
    cm = confusion_matrix(np.array(predicts), np.array(trues))
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
    plt.show()
