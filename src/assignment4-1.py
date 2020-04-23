#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal


# In[64]:


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


# In[65]:


def add_ones(dataset):
    dataset.insert(0, 'ones', [1] * len(dataset))
    return dataset


# In[66]:


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


# In[67]:


def perceptron(data, yita = 1):
    x = data.iloc[:,0:-1]
    y = data.iloc[:,-1].to_numpy()
    w = np.zeros(x.shape[1])
    
    cnt_correct = 0
    i = 0
    cnt = 0
    
    while cnt_correct < len(x) and cnt < 10 * len(data):
        predict = np.dot(w, x.iloc[i])
        if y[i] * predict <= 0:
            w = w + yita * y[i] * (x.iloc[i].to_numpy())
            cnt_correct = 0
        else:
            cnt_correct += 1
        i += 1
        i %= len(x)
        cnt += 1
    
    return w


# In[68]:


def perceptron_dual(data):
    x = data.iloc[:,0:-1]
    y = data.iloc[:,-1].to_numpy()
    alpha = np.zeros(x.shape[0])
    
    cnt_correct = 0
    i = 0
    cnt = 0
    
    while cnt_correct < len(x) and cnt < 10 * len(data):
        score = np.dot(alpha * y, np.dot(x.iloc[i], x.to_numpy().T))
        if score >= 0:
            predict = 1
        else:
            predict = -1
        if y[i] != predict:
            alpha[i] += 1
            cnt_correct = 0
        else:
            cnt_correct += 1
        i += 1
        i %= len(x)
        cnt += 1
                
    w = np.dot(alpha * y, x.to_numpy())
    return w


# In[69]:


def RBF(cur, x, gamma):
    res = np.zeros(len(x))
    for i in range(len(x)):
        row = x.iloc[i].to_numpy()
        row = row - cur.to_numpy()
        exponent = -gamma * np.dot(row, row)
        res[i] = np.exp(exponent)
    return res

def perceptron_kernel(data, kernel_func, gamma):
    if kernel_func == 1:
        return perceptron_dual(data)
    x = data.iloc[:,0:-1]
    y = data.iloc[:,-1].to_numpy()
    alpha = np.zeros(x.shape[0])
    
    cnt_correct = 0
    i = 0
    cnt = 0
    
    while cnt_correct < len(x) and cnt < 10 * len(data):
        score = np.dot(alpha * y, RBF(x.iloc[i], x, gamma))
        if score >= 0:
            predict = 1
        else:
            predict = -1
        if y[i] != predict:
            alpha[i] += 1
            cnt_correct = 0
        else:
            cnt_correct += 1
        i += 1
        i %= len(x)
        cnt += 1
                
    return alpha
        


# In[70]:


def cal_recall(predict_res, actual_labels):
    all_relevant = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == 1:
            all_relevant += 1
    all_irrelavant = len(actual_labels) - all_relevant
    
    tp1 = 0
    tpn1 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1 and actual_labels[i] == 1:
            tp1 += 1
        if predict_res[i] == -1 and actual_labels[i] == -1:
            tpn1 += 1
    avg_recall = np.mean((tp1/all_relevant, tpn1/all_irrelavant))
    return avg_recall
        
def cal_precision(predict_res, actual_labels):
    tp1 = 0
    all1 = 0
    tpn1 = 0
    alln1 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1:
            all1 += 1
            if actual_labels[i] == 1:
                tp1 += 1
        if predict_res[i] == -1:
            alln1 += 1
            if actual_labels[i] == -1:
                tpn1 += 1
    avg_precision = np.mean((tp1/all1, tpn1/alln1))
    return avg_precision


# In[71]:


def predict(test_data, w):
    x = test_data.iloc[:,0:-1]
    y = test_data.iloc[:,-1].to_numpy()
    arr = np.dot(w, x.to_numpy().T)
    
    err = 0
    predict_res = []
    for i in range(len(arr)):
        if arr[i] >= 0:
            predict_label = 1
        else:
            predict_label = -1
        predict_res.append(predict_label)
        if y[i] != predict_label:
            err += 1
    return 1 - err/len(y), predict_res, arr

def predict_kernel(test_data, train_data, alpha, gamma):
    x = test_data.iloc[:,0:-1]
    y = test_data.iloc[:,-1].to_numpy()
    x_train = train_data.iloc[:,0:-1]
    y_train = train_data.iloc[:,-1].to_numpy()
    scores = []
    
    for i in range(len(test_data)):
        score = np.dot(alpha * y_train, RBF(x.iloc[i], x_train, gamma))
        scores.append(score)   
    
    err = 0
    predict_res = []
    for i in range(len(scores)):
        if scores[i] >= 0:
            predict_label = 1
        else:
            predict_label = -1
        predict_res.append(predict_label)
        if y[i] != predict_label:
            err += 1
    return 1 - err/len(y), predict_res, scores


# In[72]:


def nested_k_folds(dataset, perceptron_type):
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
    
    i = 1
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
        means, stds, train_data = z_score(train_data)
        train_data = add_ones(train_data)

        if perceptron_type == 1:
            w = perceptron(train_data)
        elif perceptron_type == 2:
            w = perceptron_dual(train_data)
        elif perceptron_type == 3:
            w = perceptron_kernel(train_data, 1, 1)
        elif perceptron_type == 4:
            gammas = [0.25, 0.2, 0.1, 0]
            best_accuracy = 0
            best_gamma = -1
            
            for gamma in gammas:
                inner_accuracies = []
                j = 1
                for inner_train_index, inner_test_index in k_folds_split(5, train_data):
                    inner_train_data = train_data.iloc[inner_train_index]
                    alpha = perceptron_kernel(inner_train_data, 0, gamma)
                    inner_test_data = train_data.iloc[inner_test_index]
                    accuracy, predict_res, scores = predict_kernel(inner_test_data, inner_train_data, alpha, gamma)
                    inner_accuracies.append(accuracy)
                    j += 1
                mean_accuracy = np.mean(inner_accuracies)
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_gamma = gamma
            
            print(f"the best gamma for fold {i} is {best_gamma}" )
            alpha = perceptron_kernel(train_data, 0, best_gamma)
        
        y = train_data.iloc[:,-1].to_numpy()
        if perceptron_type != 4:
            accuracy_train, predict_res_train, scores_train = predict(train_data, w)
        else:
            accuracy_train, predict_res_train, scores_train = predict_kernel(train_data, train_data, alpha, best_gamma)
        recall_train = cal_recall(predict_res_train, y)
        precision_train = cal_precision(predict_res_train, y)
        train_accuracys.append(accuracy_train)
        train_recalls.append(recall_train)
        train_precisions.append(precision_train)
        
        test_data = dataset.iloc[test_index]
        test_data = z_score_with_paras(test_data, means, stds)
        test_data = add_ones(test_data)

        y = test_data.iloc[:,-1].to_numpy()
        if perceptron_type != 4:
            accuracy_test, predict_res_test, scores_test = predict(test_data, w)
        else:
            accuracy_test, predict_res_test, scores_test = predict_kernel(test_data, train_data, alpha, best_gamma)
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
    


# In[73]:


# PerceptronData
dataset = pd.read_csv('Assignment4/perceptronData.csv', header = None)
prob_name = 'Perceptron'

nested_k_folds(dataset, 1)
print("\n\n")
nested_k_folds(dataset, 2)
print("\n\n")


# In[ ]:


dataset = pd.read_csv('Assignment4/twoSpirals.csv', header = None)
prob_name = 'twoSpirals'

nested_k_folds(dataset, 3)
print("\n\n")
nested_k_folds(dataset, 4)


# In[ ]:




