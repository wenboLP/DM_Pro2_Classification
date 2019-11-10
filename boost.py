from DecisionTree import *
import math
import random

def weight_sample(data, W):
    ind = np.random.choice(len(data), size=len(data), replace=True, p=W)
    D = data[ind, :]
    return D

def cal_err(labels, labels_pre, W):
    dif = labels - labels_pre
    ind_wrong = dif*dif  # wrong labels has 1 on the positon
    err = sum(W*ind_wrong)
    return err

def w_update(W, alpha_i, labels_pre, labels):
    ind_wrong = (labels - labels_pre)*(labels - labels_pre)
    exp_to = ( (-1)*np.ones(len(labels))+2*ind_wrong ) * alpha_i
    exp_array = np.exp( exp_to )
    W_adjust = W * exp_array
    W_adjust = W_adjust/sum(W_adjust)
    return W_adjust

def boost_tree(data, labels, k=5):
    n = len(data)
    W = np.zeros(n)+1/n
    alpha_list = []
    tree_list = []

    for i in range(k):
        print('step '+str(i))
        Di = weight_sample(data, W)
        tree_i = build_tree(Di,labels)
        labels_pre, pre = tree_predict_table(tree_i, labels, data)
        err_i = cal_err(labels, labels_pre, W)
        if err_i>=0.5:
            W = np.zeros(n)+1/n
            continue
        alpha_i = 0.5 * math.log( (1-err_i)/err_i )
        W = w_update(W, alpha_i, labels_pre, labels)
        alpha_list.append(alpha_i)
        tree_list.append(tree_i)
    return alpha_list, tree_list

def boostree_pre(alpha_list, tree_list, data_sample):
    vote = 0
    for i in range(len(alpha_list)):
        vote = vote + alpha_list[i] * tree_predict(tree_list[i], data_sample)
    if vote>=0.5:
        return 1
    else:
        return 0

def boostree_pre_table(alpha_list, tree_list, labels, data):
    labels_pre = np.zeros_like(labels) 
    for i in range(len(labels_pre)):
        labels_pre[i] = boostree_pre(alpha_list, tree_list, data[i])
    pre = 1- ( np.count_nonzero(labels_pre - labels) / len(labels))
    return labels_pre, pre


if __name__ == "__main__":
    # data, labels = data_loader('project3_dataset1.txt')  # 'project3_dataset2.txt'  #(569, 30)
    data, labels = data_loader('project3_dataset2.txt')  # 'project3_dataset2.txt'  #(462, 10)

    data = data_process(data)

    alpha_list, tree_list = boost_tree(data, labels, k=30)
    alpha_list
    labels_pre, pre = boostree_pre_table(alpha_list, tree_list, labels, data)
    pre
    print('precision is '+str(pre))


#------------------------------------------------------

    tree_data = build_tree(data,labels)

    labels_pre, pre = tree_predict_table(tree_data, labels, data)
    # labels_pre
    pre
    print('precision is '+str(pre))
