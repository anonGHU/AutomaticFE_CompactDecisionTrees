#from automatic_FE.auto_by_criteria import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn import tree
import sys
import os
import ast


df_importance= pd.DataFrame(
    columns=['dataset_name','number_of_features','using_criterion','delete_used_f','importance_function',
         'method','all_features','10','20','30','40','50','60','70','80','90','100'])


df = pd.read_excel(r'D:\phd\שרת\30.5\imp\results_importance.xlsx', index_col=0)
#print(df.head)

def write_to_excel_df_importance():
    writerResults = pd.ExcelWriter(r'D:\phd\שרת\30.5\imp\import_all.xlsx')
    df_importance.to_excel(writerResults,'results')
    writerResults.save()


def get_index_from_precent(precent):
    if precent <= 0.1:
        return 1
    if precent <= 0.2:
        return 2
    if precent <= 0.3:
        return 3
    if precent <= 0.4:
        return 4
    if precent <= 0.5:
        return 5
    if precent <= 0.6:
        return 6
    if precent <= 0.7:
        return 7
    if precent <= 0.8:
        return 8
    if precent <= 0.9:
        return 9
    if precent <= 1:
        return 10
    print(precent)
    return 10


def fill_number_of_f(dic,number_of_f,precent):
    index = get_index_from_precent(precent)
    for i in range(1,index+1):
        if dic[i]==0:
            dic[i] = number_of_f
    dic[index]= number_of_f
    return dic

for index, row in df.iterrows():
    data_name = row['dataset_name']
    using_criterion = row['using_criterion']
    delete_used_f = row['delete_used_f']
    feature_importance_base = ast.literal_eval(row['feature_importance_base'])
    feature_importance_after = ast.literal_eval(row['feature_importance_after'])
    after_f_number = row['number_of_features']
    base_f_number= row['number_of_all_features']
    importance_function = row['importance_function']

    dic_number_of_f ={}
    dic_number_of_f[1] = 0
    dic_number_of_f[2] = 0
    dic_number_of_f[3] = 0
    dic_number_of_f[4] = 0
    dic_number_of_f[5] = 0
    dic_number_of_f[6] = 0
    dic_number_of_f[7] = 0
    dic_number_of_f[8] = 0
    dic_number_of_f[9] = 0
    dic_number_of_f[10] = 0

    if importance_function == 'permutation_importance':
        sum_normal =0
        for f in feature_importance_base:
            sum_normal += float(f[0])
        for ind in range(0,len(feature_importance_base)):
            feature_importance_base[ind] = (feature_importance_base[ind][0]/sum_normal,feature_importance_base[ind][1])

    sum_of_import = 0
    number_of_features = 0
    for f in feature_importance_base:
        imp = float(f[0])
        sum_of_import += imp
        number_of_features+= 1
        dic_number_of_f = fill_number_of_f(dic_number_of_f, (number_of_features)/len(feature_importance_base), sum_of_import)

    df_importance.loc[len(df_importance)] = np.array(
        [data_name, len(feature_importance_base), using_criterion, delete_used_f, importance_function,
                 'base',str(feature_importance_base), dic_number_of_f[1], dic_number_of_f[2], dic_number_of_f[3], dic_number_of_f[4],dic_number_of_f[5], dic_number_of_f[6], dic_number_of_f[7], dic_number_of_f[8], dic_number_of_f[9], dic_number_of_f[10]])

    dic_number_of_f = {}
    dic_number_of_f[1] = 0
    dic_number_of_f[2] = 0
    dic_number_of_f[3] = 0
    dic_number_of_f[4] = 0
    dic_number_of_f[5] = 0
    dic_number_of_f[6] = 0
    dic_number_of_f[7] = 0
    dic_number_of_f[8] = 0
    dic_number_of_f[9] = 0
    dic_number_of_f[10] = 0


    if importance_function == 'permutation_importance':
        sum_normal =0
        for f in feature_importance_after:
            sum_normal += float(f[0])
        for ind in range(0,len(feature_importance_after)):
            feature_importance_after[ind] = (feature_importance_after[ind][0]/sum_normal,feature_importance_after[ind][1])

    sum_of_import = 0
    number_of_features = 0
    for f in feature_importance_after:
        imp = float(f[0])
        sum_of_import += imp
        number_of_features += 1
        dic_number_of_f = fill_number_of_f(dic_number_of_f, (number_of_features) / len(feature_importance_after),
                                           sum_of_import)

    df_importance.loc[len(df_importance)] = np.array(
        [data_name, len(feature_importance_after), using_criterion, delete_used_f, importance_function,
        'after',str(feature_importance_after), dic_number_of_f[1], dic_number_of_f[2], dic_number_of_f[3], dic_number_of_f[4],
        dic_number_of_f[5], dic_number_of_f[6], dic_number_of_f[7], dic_number_of_f[8], dic_number_of_f[9],
        dic_number_of_f[10]])



    #print(res)
    #print(type(res))

write_to_excel_df_importance()
