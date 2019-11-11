import numpy as np
from data_loader import data_loader
from DecisionTree import *
from DecisionTree_prePrune import *
from randomforest import *
from boost import *


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

def metrics(y_true, y_pred):
    acc = np.mean(y_pred == y_true)
    rec = np.mean(y_pred[y_true == 1] == y_true[y_true == 1])
    pre = np.mean(y_pred[y_pred == 1] == y_true[y_pred == 1])
    f1 = 2 * rec * pre / (pre + rec)
    return acc, rec, pre, f1

if __name__ == "__main__":
    data, labels = data_loader('project3_dataset1.txt')  # 'project3_dataset2.txt'  #(569, 30)
    # data, labels = data_loader('project3_dataset2.txt')  # 'project3_dataset2.txt'  #(462, 10)

    data = data_process(data)

    # 1 decision tree
    nfolds = 10
    acc, rec, pre, f1 = 0, 0, 0, 0
    for i in range(nfolds):
        print('flod: ', i)
        x_train, y_train, x_test, y_test = fold(data, labels, i, nfolds)
        deci_tree = build_tree(x_train,y_train)
        labels_pre, pre = tree_predict_table(deci_tree, y_test, x_test)
        acc, rec, pre, f1 = metrics(y_test, labels_pre)
        acc = acc + 1.0 / nfolds * acc
        rec = rec + 1.0 / nfolds * rec
        pre = pre + 1.0 / nfolds * pre
        f1 = f1 + 1.0 / nfolds * f1
    print ('1st decision tree precision is ', ' acc ', acc, ' rec ', rec, ' pre ', pre, ' f1 ',f1)
#  1st decision tree precision is   
#  acc  0.9625  rec  0.9730769230769231  pre  0.9370370370370371  f1  0.9547169811320755


    # 2 DecisionTree_prePrune
    acc, rec, pre, f1 = 0, 0, 0, 0
    for i in range(nfolds):
        print('flod: ', i)
        x_train, y_train, x_test, y_test = fold(data, labels, i, nfolds)
        deci_tree_prune = build_tree_preprune(x_train,y_train)
        labels_pre, pre = tree_predict_table(deci_tree_prune, y_test, x_test)
        acc, rec, pre, f1 = metrics(y_test, labels_pre)
        acc = acc + 1.0 / nfolds * acc
        rec = rec + 1.0 / nfolds * rec
        pre = pre + 1.0 / nfolds * pre
        f1 = f1 + 1.0 / nfolds * f1
    print ('2st DecisionTree_prePrune precision is ', ' acc ', acc, ' rec ', rec, ' pre ', pre, ' f1 ',f1)

    # 3 random forest
    acc, rec, pre, f1 = 0, 0, 0, 0
    for i in range(nfolds):
        print('flod: ', i)
        x_train, y_train, x_test, y_test = fold(data, labels, i, nfolds)
        data_bags_list = bagging(x_train,num_bag = 10)
        forest_get = random_forest(data_bags_list, y_train )
        labels_pre, pre = forest_predict_table(forest_get, y_test, x_test)
        acc, rec, pre, f1 = metrics(y_test, labels_pre)
        acc = acc + 1.0 / nfolds * acc
        rec = rec + 1.0 / nfolds * rec
        pre = pre + 1.0 / nfolds * pre
        f1 = f1 + 1.0 / nfolds * f1
    print ('3 random forest precision is ', ' acc ', acc, ' rec ', rec, ' pre ', pre, ' f1 ',f1)
    # 3 random forest precision is   
    #  acc  0.55  rec  0.4653846153846154  pre  0.5041666666666667  f1  0.48399999999999993

    # 4 boost 
    nfolds = 10
    acc, rec, pre, f1 = 0, 0, 0, 0
    for i in range(nfolds):
        print('flod: ', i)
        x_train, y_train, x_test, y_test = fold(data, labels, i, nfolds)
        
        alpha_list, tree_list = boost_tree(x_train, y_train, k=5)
        labels_pre, pre = boostree_pre_table(alpha_list, tree_list, y_test, x_test)
        acc, rec, pre, f1 = metrics(y_test, labels_pre)
        acc = acc + 1.0 / nfolds * acc
        rec = rec + 1.0 / nfolds * rec
        pre = pre + 1.0 / nfolds * pre
        f1 = f1 + 1.0 / nfolds * f1
    print ('4 boost precision is ', ' acc ', acc, ' rec ', rec, ' pre ', pre, ' f1 ',f1)
    # 4 boost precision is acc  0.5892857142857143  rec  0.0  pre  nan  f1  nan







    # 1 decision tree
    deci_tree = build_tree(data,labels)
    labels_pre, pre = tree_predict_table(deci_tree, labels, data)
    print('1st precision is '+str(pre))

    # 2 DecisionTree_prePrune
    deci_tree_prune = build_tree_preprune(data,labels, threshold_gini = 0.0)
    labels_pre, pre = tree_predict_table(deci_tree_prune, labels, data)
    print('2nd precision is '+str(pre))

    # 3 random forest
    data_bags_list = bagging(data,num_bag = 10)
    forest_get = random_forest(data_bags_list, labels )
    labels_pre, pre = forest_predict_table(forest_get, labels, data)
    print('3rd rf precision is '+str(pre))

    # 4 boost 
    alpha_list, tree_list = boost_tree(data, labels, k=10)
    labels_pre, pre = boostree_pre_table(alpha_list, tree_list, labels, data)
    print('4th boost precision is '+str(pre))

