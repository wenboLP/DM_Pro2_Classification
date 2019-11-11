import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def fold(x, y, i, nfolds):
    num = len(y)
    num_per_fold = int(num / nfolds)
    valid_idx_s = i * num_per_fold
    valid_idx_e = valid_idx_s + num_per_fold - 1
    valid_idx = list(range(valid_idx_s, valid_idx_e+1))
    train_idx = list(set(range(num)) - set(valid_idx))
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[valid_idx]
    y_test = y[valid_idx]
    return x_train, y_train, x_test, y_test

def calc_metric(y_predict, y, metric):
    n_test = y.shape[0]
    recall = sum(y_predict[y == 1] == 1) / (sum(y == 1)+1e-8)
    precision = sum(y[y_predict == 1] == 1) / (sum(y_predict == 1)+1e-8)
    if metric == 'accuracy':
        return sum(y_predict == y) / n_test
    elif metric == 'recall':
        return recall
    elif metric == 'precision':
        return precision
    elif metric == 'F1':
        F1 = 2*recall*precision/(recall+precision+1e-8)
        return F1
    else:
        return 0

def data_plot(data, label_list, title):
    mylist = list(set(label_list))
    num_class = len(mylist)
    m = data.shape[0]
    ind_color = np.zeros((m, num_class))
    for i in range(m):
        for j in range(len(mylist)):
            if label_list[i] == mylist[j]:
                ind_color[i,j] = 1
                break
    for i in range(num_class):
        index = np.nonzero(ind_color[:, i])
        plt.scatter(data[index, 0], data[index, 1], label=mylist[i])
    plt.legend()
    plt.title(title)
    plt.show()

def data_loader(name):
    pnum = re.compile('[0-9.]')
    catnum = 0
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