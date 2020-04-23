#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import math
import random
from decimal import Decimal


# In[2]:


def z_score(dataset):
    means = dataset.mean(axis = 0, skipna = True)
    stds = dataset.std(axis = 0, skipna = True) 
    mean_list = list(means)[0:-1]
    std_list = list(stds)[0:-1]
    length = len(dataset.columns) - 1
    
    new_dataset = pd.DataFrame()
    for i in range(length):
        new_col = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
        new_dataset.insert(len(new_dataset.columns), i, new_col) 
    new_dataset.insert(len(new_dataset.columns), len(new_dataset.columns), dataset.iloc[:,-1])
    return list(means)[0:-1], list(stds)[0:-1], new_dataset
    
def z_score_with_paras(dataset, mean_list, std_list):
    length = len(dataset.columns) - 1
    
    new_dataset = pd.DataFrame()
    for i in range(length):
        new_col = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
        new_dataset.insert(len(new_dataset.columns), i, new_col) 
    new_dataset.insert(len(new_dataset.columns), len(new_dataset.columns), dataset.iloc[:,-1])
    return new_dataset


# In[3]:


def add_ones(dataset):
    dataset.insert(0, 'ones', [1] * len(dataset))
    return dataset


# In[4]:


def k_folds_split(k, dataset):
    length = len(dataset)
    piece_len = int(length / k)
    mylist = list(range(length))
    random.shuffle(mylist)
    result = []
    for i in range(k):
        test_index = mylist[i*piece_len:(i+1)*piece_len]
        train_index = mylist[0:i*piece_len] + mylist[(i+1)*piece_len:]
        result.append((train_index, test_index))
    return result


# In[5]:


def cal_accuracy(predict_res, actual_labels):
    err = 0
    for i in range(len(actual_labels)):
        if predict_res[i] != actual_labels[i]:
            err += 1
    return 1 - err/len(actual_labels)

def cal_recall(predict_res, actual_labels):
    all_relevant = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == 1:
            all_relevant += 1
    all_irrelavant = len(actual_labels) - all_relevant
    
    tp1 = 0
    tp0 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1 and actual_labels[i] == 1:
            tp1 += 1
        if predict_res[i] == 0 and actual_labels[i] == 0:
            tp0 += 1
    avg_recall = np.mean((tp1/all_relevant, tp0/all_irrelavant))
    return avg_recall
        
def cal_precision(predict_res, actual_labels):
    tp1 = 0
    all1 = 0
    tp0 = 0
    all0 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1:
            all1 += 1
            if actual_labels[i] == 1:
                tp1 += 1
        if predict_res[i] == 0:
            all0 += 1
            if actual_labels[i] == 0:
                tp0 += 1
    avg_precision = np.mean((tp1/all1, tp0/all0))
    return avg_precision


# In[6]:


def choose_C(train_data, is_optimize_accuracy):
    Cs = [(2 ** -5) * (2 ** (n - 1)) for n in range(1, 16 + 1)]
    
    best_accuracy = 0
    best_auc = 0
    best_C = -1
    
    for c in Cs:
        inner_accuracies = []
        inner_aucs = []
        for inner_train_index, inner_test_index in k_folds_split(5, train_data):
            inner_train_data = train_data.iloc[inner_train_index]
            x = inner_train_data.iloc[:,0:-1].to_numpy()
            y = inner_train_data.iloc[:,-1].to_numpy()
                    
            clf = SVC(kernel='linear', probability=True, C=c)
            clf.fit(x, y)
                    
            inner_test_data = train_data.iloc[inner_test_index]
            x = inner_test_data.iloc[:,0:-1].to_numpy()
            y = inner_test_data.iloc[:,-1].to_numpy()
                    
            if is_optimize_accuracy:
                y_predict = clf.predict(x)
                accuracy_inner_test = cal_accuracy(y_predict, y)
                inner_accuracies.append(accuracy_inner_test)
            else:
                y_prob = clf.predict_proba(x)
                classes = clf.classes_
                p_index = np.where(classes == 1)
                fpr,tpr,threshold = roc_curve(y, y_prob[:,p_index[0][0]]) 
                roc_auc = auc(fpr,tpr)
                inner_aucs.append(roc_auc)
        # end for
        if is_optimize_accuracy:
            mean_accuracy = np.mean(inner_accuracies)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_C = c
        else:
            mean_auc = np.mean(inner_aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_C = c
    # end for
    return best_C


def choose_C_gamma(train_data, is_optimize_accuracy):
    Cs = [(2 ** -5) * (2 ** (n - 1)) for n in range(1, 16 + 1)]
    gammas = [(2 ** -15) * (2 ** (n - 1)) for n in range(1, 21 + 1)]
    
    best_accuracy = 0
    best_auc = 0
    best_C = -1
    best_gamma = -1
    
    for gamma in gammas:
        for c in Cs:
            inner_accuracies = []
            inner_aucs = []
            for inner_train_index, inner_test_index in k_folds_split(5, train_data):
                inner_train_data = train_data.iloc[inner_train_index]
                x = inner_train_data.iloc[:,0:-1].to_numpy()
                y = inner_train_data.iloc[:,-1].to_numpy()
                    
                clf = SVC(kernel='rbf', probability=True, C=c, gamma=gamma)
                clf.fit(x, y)
                    
                inner_test_data = train_data.iloc[inner_test_index]
                x = inner_test_data.iloc[:,0:-1].to_numpy()
                y = inner_test_data.iloc[:,-1].to_numpy()
                    
                if is_optimize_accuracy:
                    y_predict = clf.predict(x)
                    accuracy_inner_test = cal_accuracy(y_predict, y)
                    inner_accuracies.append(accuracy_inner_test)
                else:
                    y_prob = clf.predict_proba(x)
                    classes = clf.classes_
                    p_index = np.where(classes == 1)
                    fpr,tpr,threshold = roc_curve(y, y_prob[:,p_index[0][0]]) 
                    roc_auc = auc(fpr,tpr)
                    inner_aucs.append(roc_auc)
            # end for
            if is_optimize_accuracy:
                mean_accuracy = np.mean(inner_accuracies)
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_gamma = gamma
                    best_C = c
            else:
                mean_auc = np.mean(inner_aucs)
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_gamma = gamma
                    best_C = c
    # end for
    return best_C, best_gamma


# In[7]:


def nested_k_folds(dataset, kernel_type, is_optimize_accuracy):
    result_table = {}
    result_table2 = {}
    result_table3 = {}
    for i in range(1, 11):
        result_table[str(i)] = {}
    for i in range(1, 11):
        result_table2[str(i)] = {}
    for i in range(1, 11):
        result_table3[str(i)] = {}
    result_table['mean accuracy'] = {}
    result_table['std accuracy'] = {}
    result_table2['mean recall'] = {}
    result_table2['std recall'] = {}
    result_table3['mean precision'] = {}
    result_table3['std precision'] = {}
    train_accuracys = []
    test_accuracys = []
    train_recalls = []
    test_recalls = []
    train_precisions = []
    test_precisions = []
    
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    roc_aucs = []
    
    i = 1
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
        means, stds, train_data = z_score(train_data)
        train_data = add_ones(train_data)
        
        if kernel_type == 'linear':
            best_C = choose_C(train_data, is_optimize_accuracy)
        elif kernel_type == 'rbf':
            best_C, best_gamma = choose_C_gamma(train_data, is_optimize_accuracy)
        
        if kernel_type == 'linear':
            print(f"the best C for fold {i} is {best_C}")
        elif kernel_type == 'rbf':    
            print(f"the best C for fold {i} is {best_C}, and the best gamma for fold {i} is {best_gamma}")
        
        x = train_data.iloc[:,0:-1].to_numpy()
        y = train_data.iloc[:,-1].to_numpy()
        
        if kernel_type == 'linear':
            clf = SVC(kernel='linear', probability=True, C=best_C)
        elif kernel_type == 'rbf':
            clf = SVC(kernel='rbf', probability=True, C=best_C, gamma=best_gamma)
        clf.fit(x, y)
        predict_res_train = clf.predict(x)
        
        accuracy_train = cal_accuracy(predict_res_train, y)
        recall_train = cal_recall(predict_res_train, y)
        precision_train = cal_precision(predict_res_train, y)
        train_accuracys.append(accuracy_train)
        train_recalls.append(recall_train)
        train_precisions.append(precision_train)
        
        test_data = dataset.iloc[test_index]
        test_data = z_score_with_paras(test_data, means, stds)
        test_data = add_ones(test_data)

        x = test_data.iloc[:,0:-1].to_numpy()
        y = test_data.iloc[:,-1].to_numpy()
        predict_res_test = clf.predict(x)

        y_prob = clf.predict_proba(x)
        classes = clf.classes_
        p_index = np.where(classes == 1)

        fpr,tpr,threshold = roc_curve(y, y_prob[:,p_index[0][0]]) 
        roc_auc = auc(fpr,tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_aucs.append(roc_auc)

        # plot roc curve for ith fold
        plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        accuracy_test = cal_accuracy(predict_res_test, y)
        recall_test = cal_recall(predict_res_test, y)
        precision_test = cal_precision(predict_res_test, y)
        test_accuracys.append(accuracy_test)
        test_recalls.append(recall_test)
        test_precisions.append(precision_test)
  
        result_table[str(i)]['train'] = accuracy_train
        result_table[str(i)]['test'] = accuracy_test
        result_table2[str(i)]['train'] = recall_train
        result_table2[str(i)]['test'] = recall_test
        result_table3[str(i)]['train'] = precision_train
        result_table3[str(i)]['test'] = precision_test
        i += 1
    # end for
    
    mean_train_accuracy = np.mean(train_accuracys)
    std_train_accuracy = np.std(train_accuracys, ddof=1)
    mean_test_accuracy = np.mean(test_accuracys)
    std_test_accuracy = np.std(test_accuracys, ddof=1)
    result_table['mean accuracy']['train'] = mean_train_accuracy
    result_table['std accuracy']['train'] = std_train_accuracy
    result_table['mean accuracy']['test'] = mean_test_accuracy
    result_table['std accuracy']['test'] = std_test_accuracy
    
    mean_train_recall = np.mean(train_recalls)
    std_train_recall = np.std(train_recalls, ddof=1)
    mean_test_recall = np.mean(test_recalls)
    std_test_recall = np.std(test_recalls, ddof=1)
    result_table2['mean recall']['train'] = mean_train_recall
    result_table2['std recall']['train'] = std_train_recall
    result_table2['mean recall']['test'] = mean_test_recall
    result_table2['std recall']['test'] = std_test_recall
    
    mean_train_precision = np.mean(train_precisions)
    std_train_precision = np.std(train_precisions, ddof=1)
    mean_test_precision = np.mean(test_precisions)
    std_test_precision = np.std(test_precisions, ddof=1)
    result_table3['mean precision']['train'] = mean_train_precision
    result_table3['std precision']['train'] = std_train_precision
    result_table3['mean precision']['test'] = mean_test_precision
    result_table3['std precision']['test'] = std_test_precision
    
    columns = list(range(1, 11)) + ['mean accuracy', 'std accuracy']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    columns = list(range(1, 11)) + ['mean recall', 'std recall']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table2, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    columns = list(range(1, 11)) + ['mean precision', 'std precision']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table3, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC = %0.2f)' % (mean_auc))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[8]:


def binarize(dataset):
    labels = np.unique(dataset.iloc[:,-1])
    labels.sort()
    
    datasets = []
    for label in labels:
        new_dataset = dataset.copy()
        new_dataset.drop(len(new_dataset.columns) - 1, axis=1, inplace=True)
        y_list = []
        for i in range(len(dataset)):
            if label == dataset.iloc[i,-1]:
                y_list.append(1)
            else:
                y_list.append(0)
        y_series = pd.Series(y_list)
        new_dataset.insert(len(new_dataset.columns), len(new_dataset.columns), y_series)
        datasets.append(new_dataset)
    return datasets


# In[39]:


# Wine dataset
dataset = pd.read_csv('Assignment4/wine.csv', header = None)
prob_name = 'Wine'

labels = dataset[0]
dataset.drop(0, axis=1, inplace=True)
dataset.insert(len(dataset.columns), 0, labels)
dataset.columns = list(range(len(dataset.columns)))
datasets = binarize(dataset)

for dataset in datasets:
    nested_k_folds(dataset, 'linear', True)
    nested_k_folds(dataset, 'rbf', True)


# In[ ]:




