import numpy as np
from data_loader import data_loader
# import matplotlib.pyplot as plt

# data.shape
# labels.shape

# data[0]
# data[1]

# for i in range(len(data[0])):
#     plt.plot(data[:,i])
#     plt.show()

# preprocessing, remove the last one_hot_encode column
def data_process(data):
    if len(data[0])==10:
        return data[:,0:-1]
    else:
        return data

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

def build_tree(data,labels):
    tree = {}
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
    #  0-col data to leaf
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
    # choose feature
    split_fea = None 
    hold_gini = 1
    # i is the featurn index  
    # print('len(data[0])='+str(len(data[0])))
    for i in range(len(data[0])):
        this_gini, i_split_val = cal_gini( data[:,i], labels)
        if this_gini < hold_gini:
            hold_gini = this_gini
            split_fea = i
            split_val = i_split_val
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
    tree['_left'] = build_tree(data_left,labels_left) # larger
    tree['_right'] = build_tree(data_right, labels_right) # less
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


if __name__ == "__main__":
    # data, labels = data_loader('project3_dataset1.txt')  # 'project3_dataset2.txt'  #(569, 30)
    data, labels = data_loader('project3_dataset2.txt')  # 'project3_dataset2.txt'  #(462, 10)

    data = data_process(data)

    tree_data = build_tree(data,labels)

    labels_pre, pre = tree_predict_table(tree_data, labels, data)
    # labels_pre
    pre
    print('precision is '+str(pre))

    # data1： 1.0
    # data2:  0.8982683





















