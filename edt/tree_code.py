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

import numpy as np
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_text


def learn_tree(df, result_column, names):
    y_var = df[result_column]
    X_var = df[names]
    features = list(X_var)
    no_features = len(features)
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.1, shuffle=False, stratify=None)
    model = tree(criterion='entropy', max_depth=None, max_features=no_features, splitter="best", min_samples_leaf=3)
    model.fit(X_train,y_train)
    model2 =  tree(criterion='gini', max_depth=None, max_features=no_features, splitter="best", min_samples_leaf=3)
    model2.fit(X_train,y_train)
    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
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
    tree_rules = export_text(model, feature_names=features)
    tree_rules = tree_rules.replace("row.", "")
    tree_rules = tree_rules.replace("bound", "boundary")
    tree_rules = tree_rules.replace(">  0.50", "is TRUE")
    tree_rules = tree_rules.replace("<= 0.50", "is FALSE")
    print(tree_rules)
    return accuracy, tree_rules
