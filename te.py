from DecisionTree import *
from data_loader import data_loader


def cal_gini_wholedata(lables):
    n = len(lables) 
    n1 = np.count_nonzero(lables)
    n2 = n - n1    
    gini  = 1 - (n1/n)**2 - (n2/n)**2
    return gini




from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini')
tree.fit(data, labels)
tree.score(data, labels)

a = data[0]
np.delete(a, 1)  

for i in selected_fea_ind:
    print(i)

data = data_bags_list[4]

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

# stituation that split has only one data
    if len(labels_left) == 1:
        tree['_left']['_isleaf'] = True
        tree['_left']['_class'] = labels_left[0]
        return 
    if len(labels_right) == 1:
        tree['_right']['_isleaf'] = True
        tree['_right']['_class'] = labels_right[0]
    
    # stituation that split has only one data ！！！！！！！！！！！
    if len(labels) == 1:
        tree['_isleaf'] = True
        tree['_class'] = present_label




def forest_predict(forest, data_sample):
    vote = 0
    for i in range(len(forest)):
        this_tree_pre = tree_predict(forest[0], data_sample)
        vote = vote+this_tree_pre
    if vote>=0.5*len(forest):
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
# 
