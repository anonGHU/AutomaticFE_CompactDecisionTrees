
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

from STree_ObliqueTreeBasedSVM import *

#df_importance_experiment= pd.DataFrame(
#    columns=['dataset_name','number_of_all_features','number_of_classes','dataset_size','using_criterion','delete_used_f',
#         'feature_importance_base','feature_importance_after',
#         'number_of_features','feature_base','feature_after','importance_function',
#         'accuracy_base', 'criteria_base','precision_base','recall_base','f_measure_base','roc_base','prc_base','n_leaves_base','max_depth_base','node_count_base',
#         'accuracy_after', 'criteria_after','precision_after','recall_after','f_measure_after','roc_after','prc_after','n_leaves_after','max_depth_after','node_count_after'])

df_importance= pd.DataFrame(
    columns=['dataset_name','number_of_features','using_criterion','delete_used_f','model','accuracy','importance_function',
         'method','all_features','10','20','30','40','50','60','70','80','90','100'])


print (os.path.dirname(os.path.abspath(__file__)))
start_path=(os.path.dirname(os.path.abspath(__file__)))
one_back= os.path.dirname(start_path)

dataset_name= str(sys.argv[1])

#result_path=os.path.join(one_back,  r"results\results_"+dataset_name+".txt")

index2 = 1
def write_to_excel_df_importance_experiment(ensemble,delete_used_choice,eval_result,criteria_choice):
    global index2
    writerResults = pd.ExcelWriter(os.path.join(one_back,  r"results_imp\results_importance_"+dataset_name+"_"+str(ensemble)+"_"+str(delete_used_choice)+"_"+str(eval_result)+"_"+str(criteria_choice)+"_"+str(index2)+".xlsx"))
    index2+=1
    df_importance.to_excel(writerResults,'results')
    writerResults.save()


def create_permutation_importance(rf,X_test,y_test):
    result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                    random_state=None, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()[::-1][:len(result.importances_mean)]

    permutation_importance_list = list(((imp, name)) for name, imp in zip(X_test.columns[sorted_idx], result.importances_mean[sorted_idx].T))

    #fig, ax = plt.subplots()
    #ax.boxplot(result.importances[sorted_idx].T,
    #           vert=False, labels=X_test.columns[sorted_idx])
    #ax.set_title("Permutation Importances (test set)")
    #fig.tight_layout()
    #plt.show()
    print(permutation_importance_list)
    return permutation_importance_list

def create_impurity_based_importance(rf,X_names):
    imp_impurity_list = list(((imp, name)) for name, imp in zip(X_names, rf.feature_importances_))
    new_imp_impurity_list = sorted(imp_impurity_list, key=lambda x: x[0],reverse=True )
    print(new_imp_impurity_list)
    return new_imp_impurity_list

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


def handle_import_and_save(dataset_name,using_criterion,delete_used,importance_f,f_imp_base,f_imp_after,model,acc):
    data_name = dataset_name
    using_criterion = using_criterion
    delete_used_f = delete_used
    importance_function = importance_f
    feature_importance_base = f_imp_base
    feature_importance_after = f_imp_after

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
        [data_name, len(feature_importance_base), using_criterion, delete_used_f,model,acc, importance_function,
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
        [data_name, len(feature_importance_after), using_criterion, delete_used_f, model,acc ,importance_function,
        'ours',str(feature_importance_after), dic_number_of_f[1], dic_number_of_f[2], dic_number_of_f[3], dic_number_of_f[4],
        dic_number_of_f[5], dic_number_of_f[6], dic_number_of_f[7], dic_number_of_f[8], dic_number_of_f[9],
        dic_number_of_f[10]])


def get_results(train,test,data, x_names,y_names,criteria_Function,number_of_trees=50):
    start_time = time.time()

    central_clf = RandomForestClassifier(n_estimators=number_of_trees,random_state=None)\
            .fit(data.iloc[train][x_names],np.array(data.iloc[train][y_names]))

    end_time = time.time()
    fit_time = end_time - start_time

    prediction = central_clf.predict(data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time

    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    pred_acc = acu_test
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
        #print("zero or ones")
    precision_all = metrics.precision_score(y_true, y_pred,average='weighted')
    recall_all = metrics.recall_score(y_true, y_pred,average='weighted')
    f_measure_all = metrics.f1_score(y_true, y_pred,average='weighted')
    try:
        roc_all = metrics.roc_auc_score(y_true, y_pred)
    except:
        #print("exception_roc")
        roc_all =  0
    try:
        precision_prc, recall_prc, thresholds_prc = metrics.precision_recall_curve(y_true, y_pred)
        prc_all = metrics.auc(recall_prc, precision_prc)
    except:
        #print("exception_prc - N")
        prc_all = 1
     ############## All the criterion options on cur_tree:

    max_depth = list()
    n_leaves = list()
    node_count = list()
    criterion = list()

    for tree in central_clf.estimators_:

        max_depth.append(tree.tree_.max_depth)
        n_leaves.append(tree.tree_.n_leaves)
        node_count.append(tree.tree_.node_count)
        criterion.append(criteria_Function(tree))

    max_depth_all = sum(max_depth)/len(max_depth)
    n_leaves_all = sum(n_leaves)/len(n_leaves)
    node_count_all = sum(node_count)/len(node_count)
    criterion_trees = sum(criterion)/len(criterion)

    return np.array(list((pred_acc, criterion_trees,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all)))

def importance_experiment(model,acc,db_name,criterion_name,delete_used_f,ds,new_ds_g,added_f_names,last_x_name,x_name_basic,y_name,n_estimators):

    all_x_f =x_name_basic.copy()
    for new_name in added_f_names:
        ds[new_name] = new_ds_g[new_name]
        all_x_f.append(new_name)

    #X_train_all_f, X_test_all_f, y_train_all_f, y_test_all_f = train_test_split(ds[all_x_f], ds[y_name], test_size = 0.1, random_state = None)
    #X_train_FE_f, X_test_FE_f, y_train_FE_f, y_test_FE_f = train_test_split(ds[last_x_name], ds[y_name], test_size = 0.15, random_state = None)

    if model == "RF":
        rf_all_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
                .fit(ds[x_name_basic], np.array(ds[y_name]))
        rf_FE_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
                .fit(ds[last_x_name], np.array(ds[y_name]))
    if model == "XGB":
        rf_all_f = XGBClassifier(max_depth=100000,n_estimators=n_estimators).fit(ds[x_name_basic],
                                                           np.array(ds[y_name]))
        rf_FE_f  = XGBClassifier(max_depth=100000,n_estimators=n_estimators).fit(ds[last_x_name],
                                                           np.array(ds[y_name]))
    if model == "OB":
        rf_all_f = Stree(max_depth=100000).fit(ds[x_name_basic],np.array(ds[y_name]))
        rf_FE_f = Stree(max_depth=100000).fit(ds[last_x_name],np.array(ds[y_name]))

    #rf_all_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
    #        .fit(X_train_all_f[x_name_basic], y_train_all_f)
    #rf_FE_f = RandomForestClassifier(n_estimators=n_estimators, random_state=None) \
    #        .fit(X_train_all_f[last_x_name], y_train_all_f)

    permutation_importance_all_f = create_permutation_importance(rf_all_f, ds[x_name_basic],np.array(ds[y_name]))
    if model != "OB":
        impurity_based_importance_all_f = create_impurity_based_importance(rf_all_f, x_name_basic)

    permutation_importance_FE_f = create_permutation_importance(rf_FE_f, ds[last_x_name],np.array(ds[y_name]))
    if model != "OB":
        impurity_based_importance_FE_f = create_impurity_based_importance(rf_FE_f, last_x_name)

    if model != "OB":
        handle_import_and_save(db_name, criterion_name, delete_used_f,
                           'impurity_based_importance', impurity_based_importance_all_f, impurity_based_importance_FE_f, model, acc)

    handle_import_and_save(db_name, criterion_name, delete_used_f,
                           'permutation_importance', permutation_importance_all_f, permutation_importance_FE_f, model, acc)

    write_to_excel_df_importance_experiment(model, delete_used_f, acc, criterion_name)

""" min_number_of_f = min(len(last_x_name),len(x_name_basic))
    for cur_f_number in range(1,min_number_of_f+1):
        important_f_all_permutation = list(couple[1] for couple in permutation_importance_all_f[:cur_f_number])
        important_f_all_impurity_based = list(couple[1] for couple in impurity_based_importance_all_f[:cur_f_number])
        important_FE_f_permutation = list(couple[1] for couple in permutation_importance_FE_f[:cur_f_number])
        important_FE_f_impurity_based = list(couple[1] for couple in impurity_based_importance_FE_f[:cur_f_number])

        kfold = KFold(5)
        index = 0
        results_all_permutation = 0
        results_all_impurity_based = 0
        results_FE_f_permutation = 0
        results_FE_f_impurity_based = 0
        for train, test in kfold.split(ds):
            index += 1
            arr_all_permutation = get_results(train, test, ds, important_f_all_permutation, y_name, criteria_Function, n_estimators)
            arr_all_impurity_based = get_results(train, test, ds, important_f_all_impurity_based, y_name, criteria_Function, n_estimators)
            arr_FE_f_permutation = get_results(train, test, ds, important_FE_f_permutation, y_name, criteria_Function, n_estimators)
            arr_FE_f_impurity_based = get_results(train, test, ds, important_FE_f_impurity_based, y_name, criteria_Function, n_estimators)
            results_all_permutation += arr_all_permutation
            results_all_impurity_based +=arr_all_impurity_based
            results_FE_f_permutation += arr_FE_f_permutation
            results_FE_f_impurity_based += arr_FE_f_impurity_based

        results_all_permutation /= index
        results_all_impurity_based /= index
        results_FE_f_permutation /= index
        results_FE_f_impurity_based /= index

        (acu_f_all_permutation, criterion_f_all_permutation, precision_f_all_permutation, recall_f_all_permutation, f_measure_f_all_permutation, roc_f_all_permutation, prc_f_all_permutation,
         n_leaves_f_all_permutation, max_depth_f_all_permutation, node_count_f_all_permutation) =  results_all_permutation.tolist()

        (acu_f_all_impurity_based, criterion_f_all_impurity_based, precision_f_all_impurity_based, recall_f_all_impurity_based,f_measure_f_all_impurity_based, roc_f_all_impurity_based, prc_f_all_impurity_based,
         n_leaves_f_all_impurity_based, max_depth_f_all_impurity_based, node_count_f_all_impurity_based) = results_all_impurity_based.tolist()

        (acu_FE_f_permutation, criterion_FE_f_permutation, precision_FE_f_permutation, recall_FE_f_permutation,f_measure_FE_f_permutation, roc_FE_f_permutation, prc_FE_f_permutation,
         n_leaves_FE_f_permutation, max_depth_FE_f_permutation, node_count_FE_f_permutation) = results_FE_f_permutation.tolist()

        (acu_FE_f_impurity_based, criterion_FE_f_impurity_based, precision_FE_f_impurity_based, recall_FE_f_impurity_based,f_measure_FE_f_impurity_based, roc_FE_f_impurity_based, prc_FE_f_impurity_based,
         n_leaves_FE_f_impurity_based, max_depth_FE_f_impurity_based, node_count_FE_f_impurity_based)= results_FE_f_impurity_based.tolist()

        df_importance_experiment.loc[len(df_importance_experiment)] = np.array(
            [db_name, str(f_number), str(un_class), str(ds.shape[0]), criterion_name,str(delete_used_f),
            str(impurity_based_importance_all_f),str(impurity_based_importance_FE_f),str(cur_f_number),
             str(important_f_all_impurity_based),str(important_FE_f_impurity_based),'impurity_based_importance',
             str(acu_f_all_impurity_based),str(criterion_f_all_impurity_based),
             str(precision_f_all_impurity_based), str(recall_f_all_impurity_based), str(f_measure_f_all_impurity_based), str(roc_f_all_impurity_based), str(prc_f_all_impurity_based),
             str(n_leaves_f_all_impurity_based), str(max_depth_f_all_impurity_based), str(node_count_f_all_impurity_based),
             str(acu_FE_f_impurity_based), str(criterion_FE_f_impurity_based),
             str(precision_FE_f_impurity_based), str(recall_FE_f_impurity_based), str(f_measure_FE_f_impurity_based),
             str(roc_FE_f_impurity_based), str(prc_FE_f_impurity_based), str(n_leaves_FE_f_impurity_based),
             str(max_depth_FE_f_impurity_based), str(node_count_FE_f_impurity_based)])

        df_importance_experiment.loc[len(df_importance_experiment)] = np.array(
            [db_name, str(f_number), str(un_class), str(ds.shape[0]), criterion_name, str(delete_used_f),
             str(permutation_importance_all_f), str(permutation_importance_FE_f), str(cur_f_number),
             str(important_f_all_permutation), str(important_FE_f_permutation), 'permutation_importance',
             str(acu_f_all_permutation), str(criterion_f_all_permutation),
             str(precision_f_all_permutation), str(recall_f_all_permutation), str(f_measure_f_all_permutation),
             str(roc_f_all_permutation), str(prc_f_all_permutation),
             str(n_leaves_f_all_permutation), str(max_depth_f_all_permutation), str(node_count_f_all_permutation),
             str(acu_FE_f_permutation), str(criterion_FE_f_permutation),
             str(precision_FE_f_permutation), str(recall_FE_f_permutation), str(f_measure_FE_f_permutation),
             str(roc_FE_f_permutation), str(prc_FE_f_permutation), str(n_leaves_FE_f_permutation),
             str(max_depth_FE_f_permutation), str(node_count_FE_f_permutation)])
"""


