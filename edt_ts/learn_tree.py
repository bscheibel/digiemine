# Copyright (C) 2021  Beate Scheibel
# This file is part of edtts.
#
# edtts is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# edtts is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# edtts.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_text
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

random_seed = 2
RANDOMSEED = 42

def learn_tree(df, result_column, names, result, results=None, final=False):

    y_var = df[result_column].values
    X_var = df[names]
    features = np.array(list(X_var))
    clf = ExtraTreesClassifier(n_estimators=50, random_state=RANDOMSEED)
    clf = clf.fit(X_var, y_var)
    model = SelectFromModel(clf, prefit=True, max_features=5)
    features = features[model.get_support()]
    X_var = X_var[features]
    features = list(X_var)
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.1, shuffle=False, stratify=None, random_state=RANDOMSEED)
    clf = tree(random_state=RANDOMSEED)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree(random_state=1, ccp_alpha=ccp_alpha, splitter="best")
        clf.fit(X_train, y_train)
        clfs.append(clf)
    if len(clfs) > 1:
        clfs = clfs[:-1]

    score_ = [precision_score(y_test, clf.fit(X_train, y_train).predict(X_test), average="weighted") for clf in clfs]
    if not final:
        model = clfs[-1]
    else:
        model = clfs[-1]
    model = model.fit(X_train,y_train)

    pred_model = model.predict(X_test)
    unique, counts = np.unique(y_var, return_counts=True)
    if final:
        print(np.asarray((unique, counts)).T)
        print(y_test)
        print("prediction model:", pred_model)
    n_nodes = model.tree_.node_count
    max_depth = model.tree_.max_depth
    accuracy = accuracy_score(y_test, pred_model)
    precision = precision_score(y_test, pred_model, average=None)
    used_features = []
    i = 0
    try:
        for f in model.feature_importances_:
            if f > 0:
                used_features.append(features[i])
            i = i + 1
    except:
        pass
    tree_rules = ""
    if final:
        print("Number of nodes total: ", n_nodes)
        print("Max depth: ", max_depth)
        print(('Accuracy of the model is {:.0%}'.format(accuracy)))
        print("Precision: ", precision)
        print("precision weighted: ", precision_score(y_test, pred_model, average="weighted", zero_division=0))
        print("Recall: ", recall_score(y_test, pred_model, average=None))
        print("Recall weighted: ", recall_score(y_test, pred_model, average="weighted"))
        print("Used features: ", used_features)
        tree_rules = export_text(model, feature_names=features)
        tree_rules = tree_rules.replace("row.","")
        tree_rules = tree_rules.replace("bound","boundary")
        tree_rules = tree_rules.replace(">  0.50", "is TRUE")
        tree_rules = tree_rules.replace(" <= 0.50", "is FALSE")
        print(tree_rules)
    return accuracy, used_features, tree_rules