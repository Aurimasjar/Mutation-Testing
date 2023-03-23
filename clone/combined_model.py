import numpy as np


def combine_and(predicts1, predicts2):
    assert len(predicts1) == len(predicts2)
    predicts = []
    for i in range(0, len(predicts1)):
        predicts.append(predicts1[i] and predicts2[i])
    return predicts


def combine_or(predicts1, predicts2):
    assert len(predicts1) == len(predicts2)
    predicts = []
    for i in range(0, len(predicts1)):
        predicts.append(predicts1[i] or predicts2[i])
    return predicts


def combine_and_prob(predicts1, predicts2, threshold1, threshold2):
    assert len(predicts1) == len(predicts2)
    predicts = []
    for i in range(0, len(predicts1)):
        predicts.append(np.array(predicts1[i] > threshold1 and predicts2[i] > threshold2))
    return predicts


def combine_or_prob(predicts1, predicts2, threshold1, threshold2):
    assert len(predicts1) == len(predicts2)
    predicts = []
    for i in range(0, len(predicts1)):
        predicts.append(np.array(predicts1[i] > threshold1 or predicts2[i] > threshold2))
    return predicts
