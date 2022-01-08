# AutomaticFE_CompactDecisionTrees

Implementing our algorithm requires implementing the operators used on the features, as well as the implementation of the three 
tree-based models examined mentioned in the paper under the "Tree-based models examined" section.

The operators we used can be divided into two groups. 
The mathematical binary operators, and the SVC-based binary operators as described in the paper's "Method Description" section. 
For the three tree-based models examined we used the following libraries:
For the SVC-based operators, we used sklearn.svm.SVC pyhtom library (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
The kermel was changed according to the operator we wanted to create (linear, poly, rbf, sigmoid). 
For the RF implementation we used the sklearn.ensemble.RandomForestClassifier pyhtom library (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 
Number of trees in the forest set to 1000. No maximum depth chosen, therefor nodes are expanded until all leaves are pure or until all leaves contain less than two samples. The number of features to consider when looking for the best split set to $sqrt(n_{features})$. The function to measure the quality of a split is "Gini impurity". 
For the XGB implementation we used the xgboost.XGBClassifier pyhtom library (https://xgboost.readthedocs.io/en/stable/python/python_api.html). 
Number of trees in the ensable set to 1000 as well. The booster chosen to use is tree-based models. No maximum depth chosen as well. Sampling method is uniform, each training instance has an equal probability of being selected.
For the ODT implementation we used the pypi.Stree pyhtom library (https://pypi.org/project/STree/). This implementaion of Oblique Tree classifier based on SVM nodes. The nodes are built and splitted with sklearn SVC models. Stree is a sklearn estimator. The kernel type we chose to be used in the algorithm is linear kernel, The function to measure the quality of a split is "Gini impurity".
We took into account the fact that there are different data-sets with varied attributes while creating the tree-based models. Thus in order to achieve successful results for all of the data-sets, we created a generic configuration that would fit as many data-sets as possible.
