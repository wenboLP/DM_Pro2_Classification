from DecisionTree import *
from data_loader import data_loader


def cal_gini_wholedata(lables):
    n = len(lables) 
    n1 = np.count_nonzero(lables)
    n2 = n - n1    
    gini  = 1 - (n1/n)**2 - (n2/n)**2
    return gini