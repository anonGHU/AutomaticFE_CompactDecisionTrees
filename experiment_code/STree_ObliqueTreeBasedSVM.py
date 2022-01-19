import time
from stree import Stree
import numpy as np
import pandas as pd
from sklearn import metrics
#from automatic_FE.job_wrapper import *
import os
import sys
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import re

dfbaselinePred= pd.DataFrame(
columns=['dataset_name','number_of_features','number_of_classes','dataset_size','using_criterion',
         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
         'number_of_folds','depth_max','number_of_trees_per_fold','number_of_rounds','delete_used_f', 'added_features','all_features',
         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after','method'])


print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)

dataset_name= str(sys.argv[1])

#result_path=os.path.join(one_back,  r"results\results_"+dataset_name+".txt")

index3 = 1

number_of_features_cur = 0
number_of_classes_cur = 0
using_criterion_cur=0
number_of_folds_cur=0
number_of_trees_per_fold_cur=0
delete_used_f_cur=0

pred_acc_before=0
criterion_trees_before=0
precision_all_before=0
recall_all_before=0
f_measure_all_before=0
roc_all_before=0
prc_all_before=0
n_leaves_all_before=0
max_depth_all_before=0
node_count_all_before=0
pred_acc_after=0
criterion_trees_after=0
precision_all_after=0
recall_all_after=0
f_measure_all_after=0
roc_all_after=0
prc_all_after=0
n_leaves_all_after=0
max_depth_all_after=0
node_count_all_after=0
before_quality_result=0
after_quality_result=0

def set_params(number_of_features,number_of_classes,using_criterion,number_of_folds,number_of_trees_per_fold,delete_used_f):
    global number_of_features_cur
    number_of_features_cur = number_of_features
    global number_of_classes_cur
    number_of_classes_cur = number_of_classes
    global using_criterion_cur
    using_criterion_cur = using_criterion
    global number_of_folds_cur
    number_of_folds_cur=number_of_folds
    global number_of_trees_per_fold_cur
    number_of_trees_per_fold_cur = number_of_trees_per_fold
    global delete_used_f_cur
    delete_used_f_cur =delete_used_f

def addToEXCEL_baseline(acu_test,precision,recall,f_measure,roc,prc,n_leaves,max_depth,node_count,baseline,data,depth,x_names):
    dfbaselinePred.loc[len(dfbaselinePred)] = np.array(
        [dataset_name, number_of_features_cur, number_of_classes_cur, str(data.shape[0]), using_criterion_cur, str(acu_test),
         "before_criterion",
         str(precision), str(recall), str(f_measure), str(roc), str(prc),
         str(n_leaves), str(max_depth), str(node_count),
         number_of_folds_cur, str(depth), number_of_trees_per_fold_cur, "", delete_used_f_cur,
         "",
         str(x_names), "", "",
         "", "", "",
         "", "", "",
         "", "",baseline])
    write_to_excel_df_baseline()


def write_to_excel_df_baseline():
    global index3
    writerResults = pd.ExcelWriter(os.path.join(one_back,  r"results_imp\results_baseline_"+dataset_name+"_"+str(index3)+".xlsx"))
    index3+=1
    dfbaselinePred.to_excel(writerResults,'results')
    writerResults.save()


def number_of_leafs_STree(clf):
    a = str(clf)
    num_of_leafs = (a.count("Leaf"))
    return num_of_leafs

def number_of_nodes_STree(clf):
    a = str(clf)
    num_of_nodes = (a.count("\n"))
    return num_of_nodes

def max_depth_STree(clf):
    a = str(clf)
    max_depth = (max(list((w.count("-") for w in a.split("\n")))))
    return max_depth



def number_of_leafs_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    number_of_leafs = 0
    for tree in a:
        number_of_leafs += tree.count("leaf")
    number_of_leafs /= len(a)
    return number_of_leafs

def number_of_nodes_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    num_of_nodes = 0
    for tree in a:
        num_of_nodes += tree.count("nodeid")
    num_of_nodes /= len(a)
    return num_of_nodes

def max_depth_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    max_depth_all = 0
    for tree in a:
        all_lines = tree.split('\n')
        max_depth = -1
        regex = re.compile(r"(?<=\b:)\w+")
        for line in all_lines:
            match = re.search('"depth": (\d+)', line)
            if match:
                mat = int(match.group(1))
                if max_depth < mat:
                    max_depth = mat
        # we need to add one for the depth of the leaf
        max_depth += 1
        max_depth_all += max_depth
    max_depth_all /= len(a)
    return max_depth_all


def baseline_STree_classifier_competition(train,test, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):
    if depth == None:
        cur_tree =Stree(max_depth=100000).fit(train[x_names], np.array(train[y_names]))
    else:
        cur_tree = Stree(max_depth=depth).fit(train[x_names], np.array(train[y_names]))
    start_time = time.time()
    prediction = cur_tree.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    pred_acc = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)
    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    if len(un_true) == 1 and len(un_pred) == 1:
        y_true.append(0)
        y_true.append(1)
        y_pred.append(0)
        y_pred.append(1)
        y_true.append(0)
        y_true.append(1)
        y_pred.append(1)
        y_pred.append(0)
        # print("zero or ones")
    criterion_trees = 0
    quality_result = 0
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred, average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        # print("exception_roc")
        roc_all = 0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1

    ############## All the criterion options on cur_tree:
    n_leaves_all = number_of_leafs_STree(cur_tree)
    max_depth_all = max_depth_STree(cur_tree)
    node_count_all = number_of_nodes_STree(cur_tree)
    return (np.array(list((quality_result, criterion_trees,pred_acc,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))), cur_tree)

def find_X_from_baseline_STree_train_test(train,test,x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):

    if depth == None:
        cur_tree = Stree(max_depth=100000).fit(train[x_names], train[y_names])
    else:
        cur_tree = Stree(max_depth=depth).fit(train[x_names], train[y_names])
    start_time = time.time()
    prediction = cur_tree.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    pred_acc = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)
    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    if len(un_true) == 1 and len(un_pred) == 1:
        y_true.append(0)
        y_true.append(1)
        y_pred.append(0)
        y_pred.append(1)
        y_true.append(0)
        y_true.append(1)
        y_pred.append(1)
        y_pred.append(0)
        # print("zero or ones")
    criterion_trees = 0
    quality_result_trees = 0
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred, average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        # print("exception_roc")
        roc_all = 0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1

    ############## All the criterion options on cur_tree:
    n_leaves_all = number_of_leafs_STree(cur_tree)
    max_depth_all = max_depth_STree(cur_tree)
    node_count_all = number_of_nodes_STree(cur_tree)

    return np.array(list((quality_result_trees, criterion_trees,pred_acc,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))).tolist()

def handle_baseline_results_STree(results,data,depth,x_names,baseline_from,type): #"after"/"before"
    global pred_acc_before
    global criterion_trees_before
    global precision_all_before
    global recall_all_before
    global f_measure_all_before
    global roc_all_before
    global prc_all_before
    global n_leaves_all_before
    global max_depth_all_before
    global node_count_all_before
    global pred_acc_after
    global criterion_trees_after
    global precision_all_after
    global recall_all_after
    global f_measure_all_after
    global roc_all_after
    global prc_all_after
    global n_leaves_all_after
    global max_depth_all_after
    global node_count_all_after
    global before_quality_result
    global after_quality_result
    if type == "before":
        before_quality_result, criterion_trees_before,pred_acc_before,precision_all_before, recall_all_before, f_measure_all_before, roc_all_before, prc_all_before, n_leaves_all_before, max_depth_all_before, node_count_all_before = results

    elif type == "after":
        after_quality_result, criterion_trees_after,pred_acc_after, precision_all_after, recall_all_after, f_measure_all_after, roc_all_after, prc_all_after, n_leaves_all_after, max_depth_all_after, node_count_all_after = results

    else:
        pass

    #addToEXCEL_baseline(pred_acc, precision_all, recall_all, f_measure_all, roc_all, prc_all, n_leaves_all, max_depth_all, node_count_all,
    #                        "STree - oblique "+str(baseline_from),data,depth,x_names)

def get_data_for_EXCEL():
    return before_quality_result,criterion_trees_before,pred_acc_before,after_quality_result,criterion_trees_after,pred_acc_after,precision_all_before,recall_all_before,f_measure_all_before,roc_all_before,prc_all_before,n_leaves_all_before,max_depth_all_before,node_count_all_before,precision_all_after,recall_all_after,f_measure_all_after,roc_all_after,prc_all_after, n_leaves_all_after,max_depth_all_after, node_count_all_after

def baseline_xgboost_classifier_competition(train,test,data, x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):
    if depth == None:
        cur_tree =XGBClassifier(max_depth=100000).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    else:
        cur_tree = XGBClassifier(max_depth=depth).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))

    #booster = cur_tree.get_booster()
    #parameters = 'dot'
    #num_trees=0
    #tree = booster.get_dump(
    #    fmap='',
    #    dump_format=parameters)[num_trees]
    #plot_tree(tree)
    #plt.show()
    booster = cur_tree.get_booster()
    start_time = time.time()
    prediction = cur_tree.predict(data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    pred_acc = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    y_true = pd.Series.tolist(data.iloc[test][y_names])
    y_pred = list(prediction)
    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    if len(un_true) == 1 or len(un_pred) == 1:
        y_true.append(0)
        y_true.append(1)
        y_pred.append(0)
        y_pred.append(1)
        y_true.append(0)
        y_true.append(1)
        y_pred.append(1)
        y_pred.append(0)
        # print("zero or ones")
    criterion_trees = 0
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred, average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        # print("exception_roc")
        roc_all = 0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1

    ############## All the criterion options on cur_tree:
    n_leaves_all = number_of_leafs_xgboost(booster)
    max_depth_all = max_depth_xgboost(booster)
    node_count_all = number_of_nodes_xgboost(booster)
    return (np.array(list((pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))), cur_tree)

def find_X_from_baseline_xgboost_train_test(train,test,x_names,y_names,criteria_Function,f_name=None,number_of_trees=5,depth=None):

    if depth == None:
        cur_tree = XGBClassifier(max_depth=100000).fit(train[x_names], train[y_names])
    else:
        cur_tree = XGBClassifier(max_depth=depth).fit(train[x_names], train[y_names])
    booster = cur_tree.get_booster()
    start_time = time.time()
    prediction = cur_tree.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    pred_acc = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)
    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    if len(un_true) == 1 or len(un_pred) == 1:
        y_true.append(0)
        y_true.append(1)
        y_pred.append(0)
        y_pred.append(1)
        y_true.append(0)
        y_true.append(1)
        y_pred.append(1)
        y_pred.append(0)
        # print("zero or ones")
    criterion_trees = 0
    precision_all = metrics.precision_score(y_true, y_pred, average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred, average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred, average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        # print("exception_roc")
        roc_all = 0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        # print("exception_prc - N")
        prc_all = 1

    ############## All the criterion options on cur_tree:
    n_leaves_all = number_of_leafs_xgboost(booster)
    max_depth_all = max_depth_xgboost(booster)
    node_count_all = number_of_nodes_xgboost(booster)

    return np.array(list((pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))).tolist()

def handle_baseline_results_xgboost(results,data,depth,x_names,baseline_from):
    pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all = results

    addToEXCEL_baseline(pred_acc, precision_all, recall_all, f_measure_all, roc_all, prc_all, n_leaves_all, max_depth_all, node_count_all,
                            "xgboost "+str(baseline_from),data,depth,x_names)



#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');
#ax = plt.gca()
#xlim = ax.get_xlim()
#w = svc.coef_[0]
#a = -w[0] / w[1]
#xx = np.linspace(xlim[0], xlim[1])
#yy = a * xx - svc.intercept_[0] / w[1]
#plt.plot(xx, yy)
#yy = a * xx - (svc.intercept_[0] - 1) / w[1]
#plt.plot(xx, yy, 'k--')
#yy = a * xx - (svc.intercept_[0] + 1) / w[1]
#plt.plot(xx, yy, 'k--')

 #a = str(clf)
 #   num_of_leafs = (a.count("Leaf"))
 #   num_of_nodes = (a.count("\n"))
    # arr = a.split("\n")
    # all_x = (w.count("-") for w in arr)
 #   max_depth = (max(list((w.count("-") for w in a.split("\n")))))
#print(num_of_leafs)
#print(num_of_nodes)
#print(max_depth)

#print(f"Classifier's accuracy (train): {clf.score(Xtrain, ytrain):.4f}")
#print(f"Classifier's accuracy (test) : {clf.score(Xtest, ytest):.4f}")