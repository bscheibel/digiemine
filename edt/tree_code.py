# Copyright (C) 2021  Beate Scheibel
# This file is part of edt.
#
# edt is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# edt is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# edt.  If not, see <http://www.gnu.org/licenses/>.


from sklearn.tree import _tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
#np.random.seed(0)

def calculate_remine_consistency(old_rule, new_rule):
    distance = levenshtein_distance(old_rule, new_rule)
    return distance

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def prob(val):
    return val / val.sum(axis=1).reshape(-1, 1)


def print_condition(node_indicator, leave_id, model, feature_names, feature, Xtrain, threshold, sample_id):
    print("WHEN", end=' ')
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    for n, node_id in enumerate(node_index):
        if leave_id[sample_id] == node_id:
            values = model.tree_.value[node_id]
            probs = prob(values)
            print('THEN Y={} (probability={}) (values={})'.format(
                probs.argmax(), probs.max(), values))
            continue
        if n > 0:
            print('AND ', end='')
        if Xtrain.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "= false"

        else:
            threshold_sign = "= true"
        if feature[node_id] != _tree.TREE_UNDEFINED:
            print(
                "%s %s" % (
                    feature_names[feature[node_id]],
                    threshold_sign,
                    ),
                end=' ')

def int_to_string(x):
    if(x==1):
       x ="OK"
    else:
       x = "NOK"
    return x


def learn_tree(df, result_column, names):
    y_var = df[result_column]#.astype(int)
    X_var = df[names]
    features = list(X_var)
    no_features = len(features)
    #print("No of features: " + str(no_features))
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.1, shuffle=False, stratify=None)
    model = tree(criterion='entropy', max_depth=None, max_features=no_features, splitter="best", min_samples_leaf=3) #criterion: entropy or gini, max_depth=None
    model.fit(X_train,y_train)
    model2 =  tree(criterion='gini', max_depth=None, max_features=no_features, splitter="best", min_samples_leaf=3) #criterion: entropy or gini, max_depth=None
    model2.fit(X_train,y_train)
    #pred_model2 = model2.predict(X_test)
    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
    #print(y_var.value_counts())
    #print(y_test)
    #print(pred_model)
    feature_names = df.columns[:no_features]
    target_names = (df[result_column].unique().tolist())
    unique, counts = np.unique(y_var, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print(y_test)
    print("prediction model:", pred_model)
    print(('Accuracy of the model is {:.0%}'.format(accuracy)))
    print("Recall: ", recall_score(y_test, pred_model, average=None))
    print("Recall weighted: ", recall_score(y_test, pred_model, average="weighted"))
    precision = precision_score(y_test, pred_model, average=None, zero_division=0)
    print("Precision: ", precision)
    print("precision weighted: ", precision_score(y_test, pred_model, average="weighted", zero_division=0))
    tree_rules1= export_text(model, feature_names=features)
    tree_rules2 = export_text(model2, feature_names=features)
    print("remine consistency:", calculate_remine_consistency(tree_rules1, tree_rules2))
    if(type(target_names[0])==int):
        target_names = [int_to_string(i) for i in target_names]
    plot_tree(decision_tree=model,
                feature_names=feature_names,
                class_names=str(target_names),
                filled=True)
    #plt.savefig('files/tree.png')

    node_indicator = model.decision_path(X_train)
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    leave_id = model.apply(X_train)
    tree_rules = export_text(model, feature_names=features)
    tree_rules = tree_rules.replace("row.", "")
    tree_rules = tree_rules.replace("bound", "boundary")
    tree_rules = tree_rules.replace(">  0.50", "is TRUE")
    tree_rules = tree_rules.replace("<=  0.50", "is FALSE")

    print(tree_rules)
    first = (model.predict(X_train) == 0).nonzero()[0]
    #for f in first:

    #print_condition(node_indicator, leave_id, model, features, feature, X_train, threshold, first[0])
    #print(get_rules(model2, features, target_names))
    print(get_rules(model, features, target_names))
    return accuracy, tree_rules
