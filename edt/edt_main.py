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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import sys
from edt import tree_code as tc
import re

def create_features(df, result_column, names):
    new_names = []
    names_bounds = [i for i in df.columns if "bound" in i and not "list" in i and not "last" in i]
    df_new = df[names_bounds]
    names_notbounds = [i for i in df.columns if not "bound" in i and ("min" in i or "max" in i)]
    df_new1 = df[names_notbounds]
    comparison_operators = ['<', '>', '<=', '>=']
    count = 0
    for name1 in df_new:
        if name1 not in names:
            continue
        for name2 in df_new1:
            if name1 == name2 or name2 not in names:
                continue
            for op in comparison_operators:
                expression = "row." + name1 + op + "row." + name2
                new_name = str(expression)
                df[expression] = df.swifter.apply(lambda row: (eval(expression)), axis=1)
                new_names.append(new_name)
                count += 1
    return df, new_names

def create_features_onlybounds(df, result_column, names):
    new_names = []
    names_bounds = [i for i in df.columns if "bound" in i]
    df_new = df[names_bounds]
    names_notbounds = [i for i in df.columns if not "bound" in i]
    df_new1 = df[names_notbounds]
    comparison_operators = ['<', '>', '<=', '>=']
    count = 0
    for name1 in df_new:
        if name1 not in names:
            continue
        for name2 in df_new1:
            if name1 == name2 or name2 not in names:
                continue
            for op in comparison_operators:
                expression = "row." + name1 + op + "row." + name2

                new_name = str(expression)
                df[expression] = df.swifter.apply(lambda row: (eval(expression)), axis=1)
                new_names.append(new_name)
                count += 1
    return df, new_names

def create_features_withoutbounds(df, result_column, names):
    new_names = []
    names_bounds = [i for i in df.columns]
    df_new = df[names_bounds]
    names_notbounds = [i for i in df.columns]
    df_new1 = df[names_notbounds]
    comparison_operators = ['<', '>', '<=', '>=']
    for name1 in df_new:
        if name1 not in names:
            continue
        for name2 in df_new1:
            for op in comparison_operators:
                if name1 == name2 or name2 not in names:
                    continue
                expression = "row." + name1 + op + "row." + name2
                new_name = str(expression)
                df[expression] = df.apply(lambda row: (eval(expression)), axis=1)
                new_names.append(new_name)
    return df, new_names

def define(input, result_column, combined=False):
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = pd.read_csv(input)

    df = df.fillna(False)
    num_attributes = []
    df = df.rename(columns={element: re.sub(r'\w*:', r'', element) for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace(" ", "") for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace("-", "") for element in df.columns.tolist()})

    for column in df:
        first_value = (df[column].iloc[0])
        if isinstance(first_value, (int, float, np.int64, bool)):
            num_attributes.append(column)


    num_attributes = [i for i in num_attributes if i not in result_column and i != "uuid"]
    rules = run(df,result_column, num_attributes, combined)
    return rules
def run(df, result_column, names, combined):
    if(isinstance(result_column, (int, float))):
        df[result_column] = df[result_column].map(tc.int_to_string)
    if combined:
        df, new_names = create_features(df, result_column, names)
    else:
        df, new_names = create_features_onlybounds(df, result_column, names)
    important_feat = new_names
    _, rules = tc.learn_tree(df, result_column, important_feat)
    return rules

if __name__ == "__main__":
    file = sys.argv[1]
    res = sys.argv[2]
    define(file, res)


