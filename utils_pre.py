import numpy as np
import re
from sklearn.metrics import *


def data_loader(name):
    pnum = re.compile('[0-9.]')
    with open(name, 'r') as f:
        samples = []
        labels = []
        for line in f:
            data = line.split('\t')
            labels.append(int(data[-1]))
            samples.append(data[0:-1])
    #preprocess the data
    nfeas = len(samples[0])
    nsamples = len(samples)
    data_type = np.zeros(nfeas)
    for idx, fea in enumerate(samples[-1]):
        if pnum.match(fea) != None:
            data_type[idx] = 1 #number
        else:
            data_type[idx] = 0 #category
    #find the indexes of continuous features
    cat_idx = np.where(data_type == 0)[0]
    num_idx = np.where(data_type == 1)[0]
    cat_dict = dict()
    for idx in cat_idx:
        cat_dict[idx] = set()

    for i in cat_idx:
        for n in range(nsamples):
            cat_dict[i] = cat_dict[i].union([samples[n][i]])
        cat_dict[i] = {cat:x for x, cat in enumerate(list(cat_dict[i]))}
    for idx in cat_idx:
        data_type[idx] = len(cat_dict[idx])

    labels = np.array(labels)
    data = np.zeros((nsamples,int(sum(data_type))))
    for n in range(nsamples):
        for i, idx in enumerate(num_idx):
           data[n,i] = str(samples[n][idx])
        pos = len(num_idx)
        for idx in cat_idx:
            cat_len = data_type[idx]
            p = cat_dict[idx][samples[n][idx]]
            data[n, pos+p] = 1
            pos += cat_len
    return data, labels


def metrics(y_true, y_pred):
    acc = np.mean(y_pred == y_true)
    rec = np.mean(y_pred[y_true == 1] == y_true[y_true == 1])
    pre = np.mean(y_pred[y_pred == 1] == y_true[y_pred == 1])
    f1 = 2 * rec * pre / (pre + rec)
    return acc, rec, pre, f1

