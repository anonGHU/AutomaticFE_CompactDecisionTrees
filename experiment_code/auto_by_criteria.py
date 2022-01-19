import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from STree_ObliqueTreeBasedSVM import *


#############################
#                           #
#  data sets - to evaluate  #
#                           #
#############################

#db_name = r'D:\phd\DB\Diabetes.csv'

import warnings
warnings.filterwarnings('ignore')

def number_of_leafs_STree(clf):
    a = str(clf)
    num_of_leafs = (a.count("Leaf"))
    return num_of_leafs

def normalize_results(pred,index,dataset_name,method,number_of_classes,cluster_number,dataset_size,using_model):
    pred /= index
    pred_list = pred.tolist()
    pred_list = [dataset_name, method, number_of_classes, cluster_number, dataset_size, using_model] + pred_list
    return pred_list

def encode_categorial(data):
    """
    change the category data to numbers represent the value
    :param data:
    :return:
    """
    le = LabelEncoder()
    for col in data:
        # categorial
        if data[col].dtype == 'object':
            #print("this is one"+ col)
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
        #print(col)
        #print(data[col].dtype)

def multi_class_DT_pred_old(train,test,data, x_names,y_names):
    start_time = time.time()
    central_clf = DecisionTreeClassifier(max_depth=3).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    prediction = central_clf.predict(data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    predictionOnTrain = central_clf.predict(data.iloc[train][x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
    train_size = len(train)
    test_size = len(test)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    return np.array(list((fit_time, pred_time,train_size,test_size, acu_test, acu_train, precision, recall, fscore)))
def predict_kfold_old(name,data, x_names,y_names,number_of_classes,paramK=5):
    kfold = KFold(paramK, True)
    index = 0
    size = data.shape[0]
    all_OVO_pred = 0
    all_multi_class_DT_pred =0
    for train, test in kfold.split(data):
        index += 1
        all_multi_class_DT_pred += multi_class_DT_pred(train,test,data, x_names,y_names)

    results_all_multi_class_DT_pred = normalize_results(all_multi_class_DT_pred,index,name,"decision_tree",number_of_classes,"-",size,"decision_tree")

    #dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred

##
def remove_duplicates(my_list):
  return list(dict.fromkeys(my_list))
def sort_f(f1,f2):
    f1 = str(f1).lower()
    f2 = str(f2).lower()
    if f2>f1:
        return f1+','+f2
    else:
        return f2+','+f1
def create_name(sort_stat,op,f1,f2=None):
    op_name = op.__name__
    if not sort_stat:
        if f2:
            return str(op_name)+'('+ sort_f(f1,f2) +')'
        else:
            return str(op_name)+'('+str(f1).lower()+')'
    else:
        if f2:
            return str(op_name)+'('+ str(f1).lower()+','+str(f2).lower() +')'
        else:
            return str(op_name)+'('+str(f1).lower()+')'

def prepare_new_feature(sort_stat,op, f1, f2):
    global dic_new_features
    name =create_name(sort_stat,op,f1,f2)
    if name not in dic_new_features:
        #dic_new_features[name] = (op(f1,f2),f1,f2)
        dic_new_features[name] = (op,f1,f2)
    return name



def criteria_number_of_leaves_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    number_of_leafs = 0
    for tree in a:
        number_of_leafs += tree.count("leaf")
    number_of_leafs /= len(a)
    return number_of_leafs

def criteria_number_of_nodes_xgboost(booster):
    a = (booster.get_dump(
        fmap='',
        with_stats=True,
        dump_format='json'))
    num_of_nodes = 0
    for tree in a:
        num_of_nodes += tree.count("nodeid")
    num_of_nodes /= len(a)
    return num_of_nodes

def criteria_max_depth_xgboost(booster):
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

def criteria_number_of_leaves_oblique(clf):
    a = str(clf)
    num_of_leafs = (a.count("Leaf"))
    return num_of_leafs

def criteria_number_of_nodes_oblique(clf):
    a = str(clf)
    num_of_nodes = (a.count("\n"))
    return num_of_nodes

def criteria_max_depth_oblique(clf):
    a = str(clf)
    max_depth = (max(list((w.count("-") for w in a.split("\n")))))
    return max_depth

#criteria_number_of_leaves_xgboost
#criteria_number_of_nodes_xgboost
#criteria_max_depth_xgboost

# unary , binary
def plus(f1,f2,train,test):
    return train[f1]+train[f2] ,test[f1]+test[f2]
def minus(f1,f2,train,test):
    return train[f1]-train[f2] , test[f1]+test[f2]
def divide(f1,f2,train,test):
    return pd.Series([row[f1] / row[f2] if row[f2] != 0 else 0 for index, row in train.iterrows()]) , pd.Series([row[f1] / row[f2] if row[f2] != 0 else 0 for index, row in test.iterrows()])
def multiplication(f1,f2,train,test):
    return train[f1]*train[f2] , test[f1]*test[f2]


def predict_class(xp, node ,Stree):
    if xp is None:
        return [], []
    #if node.is_leaf():
    # set a class for every sample in dataset
    #    prediction = np.full((xp.shape[0], 1), node._class)
    #    return prediction, indices
    Stree.splitter_.partition(np.array(xp), node, train=False)
    #x_u, x_d = Stree.splitter_.part(xp)
    indices = np.arange(xp.shape[0])
    i_u, i_d = Stree.splitter_.part(indices)
    xp['new'] = 1
    xp.loc[i_u,'new'] = 0
    #xp.iloc[i_u]['new'] = 0

    #df = pd.DataFrame()

    #prx_u, prin_u = predict_class(x_u, i_u, node.get_up())
    #prx_d, prin_d = predict_class(x_d, i_d, node.get_down())
    #return np.append(prx_u, prx_d), np.append(prin_u, prin_d)
    return xp


#    predict_class((data.iloc[train][x_names]).copy().reset_index(drop=True), cur_tree_linear.tree_, cur_tree_linear)
def svc_binary_linear(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train[[f1,f2]],np.array(train[y_names]))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp_train = train[[f1, f2]].copy().reset_index(drop=True)
    xp_test = test[[f1, f2]].copy().reset_index(drop=True)

    if number_of_leafs_STree_cur >1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp_train), node, train=False)
        #x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_train.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_train['new'] = 1
        if i_u is not None:
            xp_train.loc[i_u,'new'] = 0

        cur_tree_linear.splitter_.partition(np.array(xp_test), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_test.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_test['new'] = 1
        if i_u is not None:
            xp_test.loc[i_u, 'new'] = 0
    else:
        xp_train['new'] = 1
        xp_test['new'] = 1

    return xp_train['new'], xp_test['new']

def svc_prediction_linear(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train[[f1,f2]],np.array(train[y_names]))
    prediction_linear_train = cur_tree_linear.tree_._clf.predict(train[[f1,f2]])
    prediction_linear_test = cur_tree_linear.tree_._clf.predict(test[[f1,f2]])

    return pd.Series(prediction_linear_train),pd.Series(prediction_linear_test)

"""
Python's SVM implementation uses one-vs-one. That's exactly what the book is talking about.
For each pairwise comparison, we measure the decision function
The decision function is the just the regular binary SVM decision boundary
What does that to do with your question?

clf.decision_function() will give you the $D$ for each pairwise comparison
The class with the most votes win
For instance,

[[ 96.42193513 -11.13296606 111.47424538 -88.5356536 44.29272494 141.0069203 ]]

is comparing:

[AB, AC, AD, BC, BD, CD]

We label each of them by the sign. We get:

[A, C, A, C, B, C]

For instance, 96.42193513 is positive and thus A is the label for AB.

Now we have three C, C would be your prediction. If you repeat my procedure for the other two examples, you will get Python's prediction. Try it!
"""
def svc_distance_linear(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='linear').fit(train[[f1,f2]],np.array(train[y_names]))
    #prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node= cur_tree_linear.tree_

    distance_points_train =  node._clf.decision_function(train[[f1,f2]])
    xp_train = pd.DataFrame(distance_points_train)
    xp_train['new'] = xp_train.apply(lambda row: np.linalg.norm(row), axis=1)

    distance_points_test = node._clf.decision_function(test[[f1, f2]])
    xp_test = pd.DataFrame(distance_points_test)
    xp_test['new'] = xp_test.apply(lambda row: np.linalg.norm(row), axis=1)

    return xp_train['new'], xp_test['new']


def svc_binary_poly(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train[[f1, f2]], np.array(train[y_names]))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp_train = train[[f1, f2]].copy().reset_index(drop=True)
    xp_test = test[[f1, f2]].copy().reset_index(drop=True)

    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp_train), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_train.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_train['new'] = 1
        if i_u is not None:
            xp_train.loc[i_u, 'new'] = 0

        cur_tree_linear.splitter_.partition(np.array(xp_test), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_test.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_test['new'] = 1
        if i_u is not None:
            xp_test.loc[i_u, 'new'] = 0
    else:
        xp_train['new'] = 1
        xp_test['new'] = 1

    return xp_train['new'], xp_test['new']

def svc_prediction_poly(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train[[f1, f2]], np.array(train[y_names]))
    prediction_linear_train = cur_tree_linear.tree_._clf.predict(train[[f1, f2]])
    prediction_linear_test = cur_tree_linear.tree_._clf.predict(test[[f1, f2]])

    return pd.Series(prediction_linear_train), pd.Series(prediction_linear_test)

def svc_distance_poly(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='poly').fit(train[[f1, f2]], np.array(train[y_names]))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_

    distance_points_train = node._clf.decision_function(train[[f1, f2]])
    xp_train = pd.DataFrame(distance_points_train)
    xp_train['new'] = xp_train.apply(lambda row: np.linalg.norm(row), axis=1)

    distance_points_test = node._clf.decision_function(test[[f1, f2]])
    xp_test = pd.DataFrame(distance_points_test)
    xp_test['new'] = xp_test.apply(lambda row: np.linalg.norm(row), axis=1)

    return xp_train['new'], xp_test['new']

def svc_binary_rbf(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train[[f1, f2]], np.array(train[y_names]))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp_train = train[[f1, f2]].copy().reset_index(drop=True)
    xp_test = test[[f1, f2]].copy().reset_index(drop=True)

    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp_train), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_train.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_train['new'] = 1
        if i_u is not None:
            xp_train.loc[i_u, 'new'] = 0

        cur_tree_linear.splitter_.partition(np.array(xp_test), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_test.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_test['new'] = 1
        if i_u is not None:
            xp_test.loc[i_u, 'new'] = 0
    else:
        xp_train['new'] = 1
        xp_test['new'] = 1

    return xp_train['new'], xp_test['new']

def svc_prediction_rbf(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train[[f1, f2]], np.array(train[y_names]))
    prediction_linear_train = cur_tree_linear.tree_._clf.predict(train[[f1, f2]])
    prediction_linear_test = cur_tree_linear.tree_._clf.predict(test[[f1, f2]])

    return pd.Series(prediction_linear_train), pd.Series(prediction_linear_test)

def svc_distance_rbf(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='rbf').fit(train[[f1, f2]], np.array(train[y_names]))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_

    distance_points_train = node._clf.decision_function(train[[f1, f2]])
    xp_train = pd.DataFrame(distance_points_train)
    xp_train['new'] = xp_train.apply(lambda row: np.linalg.norm(row), axis=1)

    distance_points_test = node._clf.decision_function(test[[f1, f2]])
    xp_test = pd.DataFrame(distance_points_test)
    xp_test['new'] = xp_test.apply(lambda row: np.linalg.norm(row), axis=1)

    return xp_train['new'], xp_test['new']

def svc_binary_sigmoid(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train[[f1, f2]], np.array(train[y_names]))
    number_of_leafs_STree_cur = number_of_leafs_STree(cur_tree_linear)

    xp_train = train[[f1, f2]].copy().reset_index(drop=True)
    xp_test = test[[f1, f2]].copy().reset_index(drop=True)

    if number_of_leafs_STree_cur > 1:
        node = cur_tree_linear.tree_
        cur_tree_linear.splitter_.partition(np.array(xp_train), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_train.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_train['new'] = 1
        if i_u is not None:
            xp_train.loc[i_u, 'new'] = 0

        cur_tree_linear.splitter_.partition(np.array(xp_test), node, train=False)
        # x_u, x_d = Stree.splitter_.part(xp)
        indices = np.arange(xp_test.shape[0])
        i_u, i_d = cur_tree_linear.splitter_.part(indices)
        xp_test['new'] = 1
        if i_u is not None:
            xp_test.loc[i_u, 'new'] = 0
    else:
        xp_train['new'] = 1
        xp_test['new'] = 1

    return xp_train['new'], xp_test['new']

def svc_prediction_sigmoid(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train[[f1, f2]], np.array(train[y_names]))
    prediction_linear_train = cur_tree_linear.tree_._clf.predict(train[[f1, f2]])
    prediction_linear_test = cur_tree_linear.tree_._clf.predict(test[[f1, f2]])

    return pd.Series(prediction_linear_train), pd.Series(prediction_linear_test)

def svc_distance_sigmoid(f1,f2,train,test):
    cur_tree_linear = Stree(max_depth=2, kernel='sigmoid').fit(train[[f1, f2]], np.array(train[y_names]))
    # prediction_linear = cur_tree_linear.tree_._clf.predict(data[[f1,f2]])
    node = cur_tree_linear.tree_

    distance_points_train = node._clf.decision_function(train[[f1, f2]])
    xp_train = pd.DataFrame(distance_points_train)
    xp_train['new'] = xp_train.apply(lambda row: np.linalg.norm(row), axis=1)

    distance_points_test = node._clf.decision_function(test[[f1, f2]])
    xp_test = pd.DataFrame(distance_points_test)
    xp_test['new'] = xp_test.apply(lambda row: np.linalg.norm(row), axis=1)

    return xp_train['new'], xp_test['new']


def create_new_features(features_unary,features_binary):
    global new_features_names
    global dic_new_features

    new_features_names = []
    for oper in operators_binary_direction_important:
        for f1 in features_binary:
            for f2 in features_binary:
                new_col=prepare_new_feature(True,oper,f1,f2)
                #if len(dic_new_features[new_col][0].unique())>1 :
                #    new_features_names.append(new_col)
                new_features_names.append(new_col)
    for oper in operators_binary_direction_important_not_twice:
        for f1 in features_binary:
            for f2 in features_binary:
                if f1 != f2:
                    new_col=prepare_new_feature(True,oper,f1,f2)
                    #if len(dic_new_features[new_col][0].unique())>1 :
                    #    new_features_names.append(new_col)
                    new_features_names.append(new_col)
    for oper in operators_binary_direction_NOT_important:
        for f1 in features_binary:
            for f2 in features_binary:
                if f1 != f2:
                    new_col=prepare_new_feature(False,oper,f1,f2)
                    #if len(dic_new_features[new_col][0].unique())>1:
                        #new_features_names.append(new_col)
                    new_features_names.append(new_col)
    for oper in operators_unary:
        for f1 in features_unary:
            new_col =prepare_new_feature(False,oper,f1,None)
            #if len(dic_new_features[new_col][0].unique())> 1 :
                #new_features_names.append(new_col)
            new_features_names.append(new_col)


def multi_class_DT_pred(train,test,data, x_names,y_names):
    start_time = time.time()
    central_clf = DecisionTreeClassifier(max_depth=5).fit(data.iloc[train][x_names], np.array(data.iloc[train][y_names]))
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    prediction = central_clf.predict(data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    predictionOnTrain = central_clf.predict(data.iloc[train][x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
    train_size = len(train)
    test_size = len(test)
    #num_of_leaves = central_clf.tree_.n_leaves
    #precision, recall, fscore, support = metrics.precision_recall_fscore_support(
    #    pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    #return np.array(list((fit_time, pred_time,train_size,test_size, acu_test, acu_train, precision, recall, fscore)))
    #return (np.array(list((acu_test,num_of_leaves))),central_clf)
    return (np.array(list((acu_test)),central_clf))

def predict_kfold(result_quality,ensemble,data_cur, x_names,y_names,criteria_Function,f_name=None,number_of_trees=100,paramK=3,depth=None,use_baseline=None,check_features=False,delete_used_f=False):
    global data
    global X_names
    global added_f_names
    global dont_use

    kfold = KFold(paramK)
    #kfold = KFold(3)
    index = 0
    size = data_cur.shape[0]
    all_DT_pred =0
    STREE_baseline_pred =0
    XGBOOST_baseline_pred =0
    counter=0
    ones_number=0
    for train, test in kfold.split(data_cur):
        data_train = (data_cur.iloc[train]).copy().reset_index(drop=True)
        data_test = (data_cur.iloc[test]).copy().reset_index(drop=True)
        X_names_cur = features_binary.copy()
        #arr,clf = multi_class_DT_pred(train,test,data_cur, x_names,y_names)

        if check_features:
            for f_name_in in added_f_names:
                if f_name_in not in data_cur:
                    train_f, test_f = dic_new_features[f_name_in][0](dic_new_features[f_name_in][1],dic_new_features[f_name_in][2],data_train,data_test)
                        #eval(f_name_in)
                    data_train.loc[:, f_name_in] =train_f
                    data_test.loc[:, f_name_in] =test_f
                    #X_names_cur.append(f_name_in)
            pass

        if f_name:
            X_names_cur.append(f_name)
            train_f, test_f = dic_new_features[f_name][0](dic_new_features[f_name][1], dic_new_features[f_name][2],
                                                          data_train, data_test)
            # eval(f_name_in)
            data_train.loc[:, f_name] = train_f
            data_test.loc[:, f_name] = test_f

            train_un= train_f.unique()
            train_un = pd.Series(np.array(train_un * 10000).round()).unique()
            test_un = test_f.unique()
            test_un = pd.Series(np.array(test_un * 10000).round()).unique()
            if len(train_un)==1 and len(test_un)==1 :
                #print(f_name)
                counter+=1
                continue

            #cur_data.loc[:, f_name] = dic_new_features[f_name][0]
            if delete_used_f:
                data_train = data_train.drop([dic_new_features[f_name][1]], axis=1)
                data_test = data_test.drop([dic_new_features[f_name][1]], axis=1)
                X_names_cur.remove(dic_new_features[f_name][1])
                if dic_new_features[f_name][1] != dic_new_features[f_name][2]:
                    data_train = data_train.drop([dic_new_features[f_name][2]], axis=1)
                    data_test = data_test.drop([dic_new_features[f_name][2]], axis=1)
                    X_names_cur.remove(dic_new_features[f_name][2])

        arr,clf,ones = find_X_from_RF(result_quality,ensemble,data_train,data_test,X_names_cur,y_names,criteria_Function,f_name,number_of_trees,depth)
        all_DT_pred += arr
        ones_number = ones_number + ones
        index += 1
        if use_baseline:
            arr_baseline1, clf = baseline_STree_classifier_competition(data_train,data_test, x_names,y_names,criteria_Function,f_name,number_of_trees,depth)
            STREE_baseline_pred += arr_baseline1
            #arr_baseline2, clf = baseline_xgboost_classifier_competition(train, test, data_cur, x_names, y_names,criteria_Function, f_name, number_of_trees, depth)
            #XGBOOST_baseline_pred += arr_baseline2


    #results_all_multi_class_DT_pred = normalize_results(all_multi_class_DT_pred,index,name,"decision_tree",number_of_classes,"-",size,"decision_tree")
    if f_name and (counter >0 or ones_number>0):
        dont_use.append(f_name)
        return (np.array(list((0, sys.maxsize,0, 0, 0, 0, 0, 0,
                               0, 0, 0))), None)

    all_DT_pred /= index
    all_pred_measures_list = all_DT_pred.tolist()

    if use_baseline:
        STREE_baseline_pred /= index
        STREE_baseline_pred_list = STREE_baseline_pred.tolist()
        handle_baseline_results_STree(STREE_baseline_pred_list, data_cur.copy(), depth, x_names, "train",use_baseline)
        #XGBOOST_baseline_pred /= index
        #XGBOOST_baseline_pred_list = XGBOOST_baseline_pred.tolist()
        #handle_baseline_results_xgboost(XGBOOST_baseline_pred_list, data_cur.copy(), depth, x_names, "train")



    return all_pred_measures_list,clf
    #dfAllPred.loc[len(dfAllPred)] = results_all_multi_class_DT_pred


def all_words_in_string(string):
    global added_f_names
    for word in added_f_names:
        if word not in string:
            return False
    return True


def find_X_from_RF(result_quality,ensemble,train,test, x_names,y_names,criteria_Function,f_name=None,number_of_trees=100,depth=None):
    start_time = time.time()

    if ensemble == "RF":
        if depth == None:
            central_clf = RandomForestClassifier(n_estimators=number_of_trees, random_state=None) \
                .fit(train[x_names], np.array(train[y_names]))
        else:
            central_clf = RandomForestClassifier(n_estimators=number_of_trees, max_depth=depth, random_state=None) \
                .fit(train[x_names], np.array(train[y_names]))
    if ensemble == "XGB":
        if depth == None:
            central_clf = XGBClassifier(max_depth=100000,n_estimators=number_of_trees).fit(train[x_names],
                                                           np.array(train[y_names]))
        else:
            central_clf = XGBClassifier(max_depth=depth,n_estimators=number_of_trees).fit(train[x_names],
                                                          np.array(train[y_names]))
    if ensemble == "OB":
        if depth == None:
            central_clf = Stree(max_depth=100000).fit(train[x_names],
                                                           np.array(train[y_names]))
        else:
            central_clf = Stree(max_depth=depth).fit(train[x_names],
                                                          np.array(train[y_names]))

    end_time = time.time()
    fit_time = end_time - start_time

    '''
    correct_trees = list()
    index_trees=0
    for cur_tree in central_clf.estimators_:
        if index_trees == number_of_trees:
            break
        if f_name:
            text_representation = tree.export_text(cur_tree, feature_names=x_names)
            if f_name in text_representation and all_words_in_string(text_representation):
                correct_trees.append(cur_tree)
                index_trees += 1
        else:
            correct_trees.append(cur_tree)
            index_trees += 1
    '''

    criterion_trees=0
    n_leaves_all =0
    max_depth_all=0
    node_count_all=0
    result_quality_cur = 0

    #if index_trees != number_of_trees:          ####check this####
    #    global criterion_base
    #    pred_acc =0
    #    criterion_trees = criterion_base
    #else:
    start_time = time.time()
    prediction = central_clf.predict(test[x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    acu_test = metrics.accuracy_score(pd.Series.tolist(test[y_names]), prediction)
    pred_acc = acu_test
    #confusion_all += confusion_matrix(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    y_true = pd.Series.tolist(test[y_names])
    y_pred = list(prediction)

    result_quality_cur = result_quality(y_true,y_pred)

    un_true, _ = np.unique(y_true, return_counts=True)
    un_pred, _ = np.unique(y_pred, return_counts=True)
    only_one = 0
    if len(un_true) == 1 or len(un_pred) == 1:
        #print("bad "+str(x_names))
        only_one = 1
    #if len(un_true) == 1 and len(un_pred) == 1:
        #print("both "+str(x_names))
    #    y_true.append(0)
    #    y_true.append(1)
    #    y_pred.append(0)
    #    y_pred.append(1)
    #    y_true.append(0)
    #    y_true.append(1)
    #    y_pred.append(1)
    #    y_pred.append(0)
        #print("zero or ones")
    precision_all = metrics.precision_score(y_true, y_pred,average='weighted',zero_division=0)
    recall_all = metrics.recall_score(y_true, y_pred,average='weighted',zero_division=0)
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

    if ensemble == "RF":
        for cur_tree in central_clf.estimators_:
            criterion_trees += criteria_Function(cur_tree)
            n_leaves_all += cur_tree.tree_.n_leaves
            max_depth_all += cur_tree.tree_.max_depth
            node_count_all += cur_tree.tree_.node_count
        n_leaves_all /= number_of_trees
        max_depth_all /= number_of_trees
        node_count_all /= number_of_trees
        criterion_trees /= number_of_trees

    if ensemble=="XGB":
        booster = central_clf.get_booster()
        criterion_trees = criteria_Function(booster)
        n_leaves_all = criteria_number_of_leaves_xgboost(booster)
        max_depth_all = criteria_max_depth_xgboost(booster)
        node_count_all = criteria_number_of_nodes_xgboost(booster)

    if ensemble == "OB":
        criterion_trees = criteria_Function(central_clf)
        n_leaves_all = criteria_number_of_leaves_oblique(central_clf)
        max_depth_all = criteria_max_depth_oblique(central_clf)
        node_count_all = criteria_number_of_nodes_oblique(central_clf)
        size = sys.getsizeof(central_clf)

    return (np.array(list((result_quality_cur, criterion_trees,pred_acc,precision_all,recall_all,f_measure_all,roc_all,prc_all,n_leaves_all,max_depth_all,node_count_all))), central_clf,only_one)


def choose_best_feature(result_quality,ensemble,base_quality_results,base_criterion,criteria_Function,depth=None,number_of_kFolds=3,number_of_trees_per_fold=100,delete_used_f=False):
    best_critrion ={}
    global data
    global X_names

    for name in new_features_names:
        if name not in added_f_names and name not in dont_use:
            cur_data = data.copy()
            #new_pred,clf = predict_kfold(cur_data, X_names_cur, y_names,criteria_Function,'new_f',number_of_trees_per_fold,number_of_kFolds,depth)
            #(data_cur, x_names,y_names,criteria_Function,f_name=None,number_of_trees=100,paramK=3,depth=None,use_baseline=False,check_features=False,delete_used_f=False)
            new_pred,clf = predict_kfold(result_quality,ensemble,cur_data, X_names.copy(), y_names,criteria_Function,name,number_of_trees_per_fold,number_of_kFolds,depth,use_baseline=None,check_features=True,delete_used_f=delete_used_f)
            new_quality_results = new_pred[0]
            new_criterion = new_pred[1]
            if new_criterion == sys.maxsize:
                pass
                #print(name)
            if new_quality_results > (base_quality_results-0.03) and new_criterion < base_criterion:
                best_critrion[name] = new_pred

    if len(best_critrion)==0:
        return None,(None,None, None, None, None, None, None, None,None, None, None)
    #min(value[1] for key,value in best_critrion.items())
    best_f = list(filter(None,[(key1,value1) if value1[1]==(min(value[1] for key,value in best_critrion.items())) else None for key1,value1 in best_critrion.items()]))[0]
    #data.loc[:, best_f[0]] = dic_new_features[best_f[0]][0]
    #X_names.append(best_f[0])
    #added_f_names.append(best_f[0])
    #features_binary.append(best_f[0])
    #if delete_used_f:
    #    data = data.drop([dic_new_features[best_f[0]][1]], axis=1)
    #    X_names.remove(dic_new_features[best_f[0]][1])
    #    if dic_new_features[best_f[0]][1] != dic_new_features[best_f[0]][2]:
    #        data = data.drop([dic_new_features[best_f[0]][2]], axis=1)
    #        X_names.remove(dic_new_features[best_f[0]][2])
    return best_f


def auto_F_E(result_quality,ensemble,number_of_kFolds,number_of_trees_per_fold,data_cur,X_names_cur,y_names_cur,features_unary_data,features_binary_data,depth,result_path,criteria_Function,delete_used_f=False):
    #print(str(delete_used_f))
    global new_features_names
    global data
    global X_names
    global y_names
    global dic_new_features
    global added_f_names
    global basic_data

    global features_binary
    features_binary = features_binary_data.copy()
    global features_unary
    features_unary = features_unary_data.copy()

    basic_data = data_cur.copy()
    dic_new_features = {}
    new_features_names = []
    added_f_names = []
    data = data_cur
    X_names =X_names_cur.copy()
    #print(X_names)
    y_names=y_names_cur
   # base_pred,clf_list_base = \
    (quality_result_base, base_criterion,base_acu_test, precision_base, recall_base, f_measure_base, roc_base, prc_base,
         n_leaves_base, max_depth_base, node_count_base),clf = \
        predict_kfold(result_quality,ensemble,data, X_names, y_names,criteria_Function,None,number_of_trees_per_fold,number_of_kFolds,depth,"before")
    #base_acu_test = base_pred[0]
    #base_criterion = base_pred[1]

    global criterion_base
    criterion_base = base_criterion

    precision_last=0
    recall_last=0
    f_measure_last=0
    roc_last=0
    prc_last=0
    n_leaves_last=0
    max_depth_last=0
    node_count_last=0

    #print(base_acu_test)
    #print(base_criterion)
    quality_result_test = quality_result_base
    criterion_cur = base_criterion
    last_acu_test=base_acu_test
    last_quality_result=quality_result_base
    f = open(result_path,"a")
    f.write("data-set: \n")
    f.write(str(delete_used_f)+ "\n")
    f.write(str(quality_result_base)+" \n")
    f.write(str(base_criterion)+" \n")
    rounds = 0
    last_criterion_cur=-1
    for num in range (5):
        create_new_features(features_unary,features_binary)
        new_features_names = list(dict.fromkeys(new_features_names))
        #print(new_features_names)
        f.write("all new f: "+ str(new_features_names) + " \n")
        new_acc = max(quality_result_base,quality_result_test)  ######## what do you think about this?
        new_f_name,(quality_result_test,criterion_cur,acu_test, precision_cur, recall_cur, f_measure_cur, roc_cur, prc_cur,
                              n_leaves_cur, max_depth_cur, node_count_cur) = choose_best_feature(result_quality,ensemble,new_acc,criterion_cur,criteria_Function,depth,number_of_kFolds,number_of_trees_per_fold,delete_used_f)
#        return (np.array(list((pred_acc, criterion_trees, precision_all, recall_all, f_measure_all, roc_all, prc_all,
#                              n_leaves_all, max_depth_all, node_count_all))), correct_trees)

        #print(X_names)
        f.write("X_names "+  str(new_f_name) + " \n")
        if new_f_name == None:
            break
        rounds=rounds+1
        added_f_names.append(new_f_name)
        features_binary.append(new_f_name)
        if delete_used_f:
            features_binary.remove(dic_new_features[new_f_name][1])
            if dic_new_features[new_f_name][1] != dic_new_features[new_f_name][2]:
                features_binary.remove(dic_new_features[new_f_name][2])
        #        data = data.drop([dic_new_features[best_f[0]][2]], axis=1)
        #        X_names.remove(dic_new_features[best_f[0]][2])
        #if delete_used_f:  #do something in the future
        #    pass
        #features_binary.append(new_f_name)
        #features_binary = features_binary.append(new_f_name)
        #print(added_f_names)
        f.write("added: "+ str(added_f_names)+" \n")
        #print(acu_test)
        #print(criterion_cur)
        last_quality_result = quality_result_test
        last_acu_test = acu_test
        last_criterion_cur = criterion_cur

        precision_last=precision_cur
        recall_last=recall_cur
        f_measure_last=f_measure_cur
        roc_last =roc_cur
        prc_last =prc_cur
        n_leaves_last =n_leaves_cur
        max_depth_last =max_depth_cur
        node_count_last =node_count_cur

        f.write(str(acu_test)+" \n")
        f.write(str(criterion_cur)+" \n")
    f.close()

    #def predict_kfold(data_cur, x_names,y_names,criteria_Function,f_name=None,number_of_trees=100,paramK=3,depth=None,use_baseline=False,check_features=False,delete_used_f=False):
    arr2, clf = \
        predict_kfold(result_quality,ensemble,data.copy(), features_binary, y_names, criteria_Function, None, number_of_trees_per_fold, number_of_kFolds,
                      depth, "after",check_features=True,delete_used_f=delete_used_f)

    return quality_result_base,base_criterion,base_acu_test,rounds,added_f_names,last_quality_result,last_criterion_cur,last_acu_test, precision_base, recall_base, f_measure_base, roc_base, prc_base,n_leaves_base, max_depth_base, node_count_base,precision_last, recall_last, f_measure_last, roc_last, prc_last,n_leaves_last, max_depth_last, node_count_last,features_binary


#before_acu_test, before_criterion, rounds, added_f_names, acu_test, criterion_after, precision_before, recall_before, f_measure_before, roc_before, prc_before, n_leaves_before, max_depth_before, node_count_before, precision_after, recall_after, f_measure_after, roc_after, prc_after, n_leaves_after, max_depth_after, node_count_after, last_x_name = \
#    auto_F_E(number_of_kFolds, number_of_trees_per_fold, data_chosen.copy(), X_names_ds, y_names, features_unary,
#             features_binary, cur_depth, result_path, criterion_function, delete_used_f)

dic_new_features={}
new_features_names = []
added_f_names = []
dont_use=[]
data = None
basic_data = None
X_names = None
y_names = None
criterion_base = 0
features_binary =[]
features_unary=[]
operators_binary_direction_important_not_twice= []  #minus
operators_binary_direction_important= [] #divide
#operators_binary_direction_important= []
operators_binary_direction_NOT_important= [#multiplication,   #plus
svc_binary_linear,svc_prediction_linear,svc_distance_linear,
svc_binary_poly,svc_prediction_poly,svc_distance_poly,
svc_binary_rbf,svc_prediction_rbf,svc_distance_rbf,
svc_binary_sigmoid,svc_prediction_sigmoid,svc_distance_sigmoid]
#operators_binary_direction_NOT_important= [plus]
operators_unary=[]

# for key, value in my_dict.items():


