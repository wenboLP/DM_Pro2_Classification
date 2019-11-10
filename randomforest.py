import numpy as np
from data_loader import data_loader
# import matplotlib.pyplot as plt



# preprocessing, remove the last one_hot_encode column
def data_process(data):
    if len(data[0])==10:
        return data[:,0:-1]
    else:
        return data

# bagging/ get new data sets
# return a list "data_bags_list", each entry of the list is a new data set
def bagging(data,num_bag = 10): 
    n = len(data)
    ind = list(range(n))
    data_bags_list = []
    for i in range(num_bag):
        ind_i = np.random.choice(ind, n, replace=True)
        data_i = data[ind_i,:]
        data_bags_list.append(data_i)
    return data_bags_list


def cal_gini( data_col, labels ):
    # data_col = data[:,0]  # one dimension np.array
    points = np.unique(data_col)
    best_gini = 1
    best_point = None
    if len(points) == 1:  # ！！！！！！！！！！！！！
        p = sum(labels) / len(labels)
        this_gini = 2*p*(1-p)
        best_gini = this_gini
        best_point = points[0]
    for point_val in points[1:]:  # for two number [0,1], still work
        data_less = labels[data_col<point_val]
        data_large = labels[data_col>=point_val]
        p_less = sum(data_less) / len(data_less)
        p_large = sum(data_large) / len(data_large)
        point_gini = len(data_less)/len(labels) * (2*p_less*(1-p_less)) + len(data_large)/len(labels) * (2*p_large*(1-p_large))
        # print('point_gini = '+str(point_gini))
        if point_gini<best_gini:
            best_gini = point_gini
            best_point = point_val
    return best_gini, best_point

def cal_gini_wholedata(labels):
    n = len(labels) 
    n1 = np.count_nonzero(labels)
    n2 = n - n1    
    gini  = 1 - (n1/n)**2 - (n2/n)**2
    # print(gini)
    return gini
    
def build_tree_rf(data,labels, threshold_gini = 0.0):
    tree = {}
    mini_gini = threshold_gini
    tree['_isleaf'] = False
    tree['_class'] = None
    tree['_split_fea'] = None
    tree['_split_fea_val'] = None
    tree['_left'] = {}  # larger  # true for >=
    tree['_right'] = {}  # less
    # present most label
    if sum(labels) >= 0.5*len(labels):
        present_label = 1
    else:
        present_label = 0
    # prePrune, check if data's gini is small enough
    if cal_gini_wholedata(labels) < mini_gini:
        tree['_isleaf'] = True
        tree['_class'] = present_label
        return tree
    #  0-col data to leaf (feature is used out)
    if len(data[0]) == 0 :
        tree['_isleaf'] = True
        tree['_class'] = present_label
        return tree
    # stituation that split has only one data !!!!!!!!!!!!!!
    if len(labels) == 1:
        tree['_isleaf'] = True
        tree['_class'] = present_label
        return tree
    # judge if it is leaf
    if sum(labels) == 0 or sum(labels) == len(labels):  # all lable 0 or 1
        tree['_isleaf'] = True
        tree['_class'] = labels[0]
        return tree        
    # choose feature / from random feature set (sqrt(n) random features)
    split_fea = None 
    hold_gini = 1
    whole_fea_ind = np.array( list(range(len(data[0]))) )
        # key: select subset
    selected_fea_ind = np.random.choice(whole_fea_ind, int(np.ceil(np.sqrt(len(data[0])))), replace=False)
    # print('selected_fea_ind = '+str(selected_fea_ind))
    for i in selected_fea_ind:
        this_gini, i_split_val = cal_gini( data[:,i], labels)
        if this_gini < hold_gini:
            hold_gini = this_gini
            split_fea = i
            split_val = i_split_val
    # print('select :' + str(i)+'  split_val='+ str(split_val) )
    # set tree
    tree['_split_fea'] = split_fea
    tree['_split_fea_val'] = split_val
    # split tree
    rows_left = data[:,split_fea] >= split_val
    data_left = data[rows_left,:]
    labels_left = labels[rows_left]
    rows_right = data[:,split_fea] < split_val
    data_right = data[rows_right,:]
    labels_right = labels[rows_right]
    # check null split 
    if len(labels_left) == 0:
        tree['_isleaf'] = True
        tree['_class'] = present_label
        return tree   
    if len(labels_right) == 0:
        tree['_isleaf'] = True
        tree['_class'] = present_label
        return tree  
    # delete used feature col
    data_left = np.delete(data_left, split_fea, axis=1)
    data_right = np.delete(data_right, split_fea, axis=1)    
    tree['_left'] = build_tree_rf(data_left,labels_left) # larger
    tree['_right'] = build_tree_rf(data_right, labels_right) # less
    return tree

def tree_predict(tree, data_sample):
    if tree['_isleaf'] == True:
        return tree['_class'] 
    fea = tree['_split_fea'] 
    val = tree['_split_fea_val']
    sample_val = data_sample[fea]
    data_sample_new = np.delete(data_sample, fea)  
    if sample_val >= val:
        return tree_predict(tree['_left'], data_sample_new)
    else:
        return tree_predict(tree['_right'], data_sample_new)

def tree_predict_table(tree, labels, data):
    labels_pre = np.zeros_like(labels) 
    for i in range(len(labels_pre)):
        labels_pre[i] = tree_predict(tree, data[i])
    pre = 1- ( np.count_nonzero(labels_pre - labels) / len(labels))
    return labels_pre, pre

def random_forest(data_bags_list, labels ):
    forest = []
    n_bag = len(data_bags_list)
    for i in range(n_bag):
        print('this tree '+str(i))
        tree_i = build_tree_rf(data_bags_list[i],labels, threshold_gini = 0.0)
        forest.append(tree_i)
    return forest

def forest_predict(forest, data_sample):
    vote = 0
    for i in range(len(forest)):
        this_tree_pre = tree_predict(forest[0], data_sample)
        vote = vote+this_tree_pre
    if vote>=0.5**len(forest):
        return 1
    else:
        return 0
# forest_predict(forest_get, data[0])

def forest_predict_table(forest, labels, data):
    labels_pre = np.zeros_like(labels) 
    for i in range(len(labels_pre)):
        labels_pre[i] = forest_predict(forest, data[i])
    pre = 1- ( np.count_nonzero(labels_pre - labels) / len(labels))
    return labels_pre, pre



if __name__ == "__main__":
    data, labels = data_loader('project3_dataset1.txt')  # 'project3_dataset1.txt'  #(569, 30)
    # data, labels = data_loader('project3_dataset2.txt')  # 'project3_dataset2.txt'  #(462, 10)

    data = data_process(data)

    # tree_i = build_tree_rf(data, labels, threshold_gini = 0.0)

    data_bags_list = bagging(data,num_bag = 10)
    forest_get = random_forest(data_bags_list, labels )
    labels_pre, pre = forest_predict_table(forest_get, labels, data)
    pre
    print('rf+bagging precision is '+str(pre))
    #      data1        data2
    #     0.5184      0.6428  (bag=10)
    #                 0.5865  (bag=20)

    forest_get_o = [data, data, data]
    forest_get_o = random_forest(forest_get_o, labels )
    labels_pre, pre = forest_predict_table(forest_get_o, labels, data)
    pre
    print('fr+odata precision is '+str(pre))

    #      data1        data2
    #       1          0.9069/ 0.89826/ 0.9177























