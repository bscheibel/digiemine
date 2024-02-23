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

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tsfresh import extract_features, select_features
from edt_ts import learn_tree as tc
from edt import edt_main as em
import swifter

def prepare_dataset(df, id, variable_interest):
    df = df.groupby(id).agg({list, "last"})
    df.columns = [' '.join(col).replace(" ", "") for col in df.columns]
    df[variable_interest + 'list'] = df[variable_interest + 'list'].swifter.apply(np.array)
    X = []
    values = df[[variable_interest + 'list']].copy()
    for v in values[variable_interest + 'list']:
        v = v[~np.isnan(v)]
        X.append(v)
    df[variable_interest + 'list'] = X
    df = df.dropna()
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    return df, colnames_numerics_only


def generate_interval_features(df, n, variable_interest):
    new_names = ["segment" + str(inter) for inter in range(1, n + 1)]

    def split(x, n):
        arrayss = np.array_split(x, n)
        return arrayss

    df[new_names] = (df.swifter.apply(lambda row: split(row[variable_interest], n), axis=1, result_type="expand"))
    for name in new_names:
        df[name + "_min"] = df.swifter.apply(lambda row: min(row[name], default=row[name]), axis=1, dtype)
        df[name + "_max"] = df.swifter.apply(lambda row: max(row[name], default=row[name]), axis=1)
        df[name + "_mean"] = df.swifter.apply(lambda row: np.mean(row[name]), axis=1)
        df[name + "_wthavg"] = df.swifter.apply(lambda row: np.average(row[name]), axis=1)
        df[name + "_sum"] = df.swifter.apply(lambda row: sum(row[name]), axis=1)
        df[name + "_std"] = df.swifter.apply(lambda row: np.std(row[name]), axis=1)
    return df


def generate_global_features(df, y_var, id, interest):
    # https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    extracted_features = extract_features(df, column_id=id, column_value=interest)
    extracted_features = extracted_features.dropna(axis=1)
    try:
        features_filtered = select_features(extracted_features, y_var)
    except:
        features_filtered = extracted_features

    features_filtered[id] = features_filtered.index
    features_filtered.reset_index()
    return features_filtered


def create_latent_variables(candidate_thresholds, df, interest_variable):
    new_names = []
    for c in candidate_thresholds:
        data = c[0]
        counter = c[1]
        expression = interest_variable + "list.count(" + str(data) + ")>=" + str(counter)
        new_name = str(expression)
        expression = "sum(i >=" + str(data) + "for i in list(row." + interest_variable + "list))" + ">=" + str(counter)
        df[new_name] = df.swifter.apply(lambda row: (eval(expression)), axis=1)
        new_names.append(new_name)
    return df, new_names


def get_distribution(array_ok, array_nok):
    frequencies = []
    for ok in array_ok:
        (unique, counts) = np.unique(ok, return_counts=True)
        f = np.asarray((unique, counts)).T
        frequencies.append(f)
    frequencies_average_ok = dict()
    for f in frequencies:
        for value, counter in f:
            if (value, counter) not in frequencies_average_ok:
                frequencies_average_ok[value, counter] = 1
            if (value, counter) in frequencies_average_ok:
                frequencies_average_ok[value, counter] = frequencies_average_ok[value, counter] + 1

    frequencies = []
    for nok in array_nok:
        (unique, counts) = np.unique(nok, return_counts=True)
        f = np.asarray((unique, counts)).T
        frequencies.append(f)
    frequencies_average_nok = dict()
    for f in frequencies:
        for value, counter in f:
            if (value, counter) not in frequencies_average_nok:
                frequencies_average_nok[value, counter] = 1
            if (value, counter) in frequencies_average_nok:
                frequencies_average_nok[value, counter] = frequencies_average_nok[value, counter] + 1

    diff = []
    for f in frequencies_average_nok.keys():
        if f not in frequencies_average_ok:
            diff.append(f)
    return diff


def get_candidate_variables(df, id):
    df = df.sort_values(by=[id, 'time:timestamp'])
    variable_names = list(df)
    temp_var = []
    cand = []
    reoccuring_variables = []
    uuid = df.iloc[0, 0]
    for column, row in df.iterrows():
        for var in variable_names:
            if var == id:
                if row[var] == uuid:
                    continue
                else:
                    uuid = row[var]
                    reoccuring_variables.extend(cand)
                    temp_var = []
            else:
                if not (isinstance(row[var], (int, float, np.int64, bool))):
                    continue
                else:
                    if var in temp_var:
                        cand.append(var)
                    else:
                        temp_var.append(var)
    reoccuring_variables = set(reoccuring_variables)
    constant_variables = [c for c in reoccuring_variables if len(set(df[c])) == 1]
    candidate_var = [c for c in reoccuring_variables if c not in constant_variables]
    return candidate_var


def sort_array_ok_nok(df, id, variable_result, variable_interest, result_column):
    candidates = dict()
    uuids = set(df[id])
    uuids_complete = []
    array_ok = []
    array_nok = []
    for uuid in uuids:
        subsetDataFrame = df[df[id] == uuid]
        values = ((subsetDataFrame[variable_interest].to_numpy()))
        values = [v for v in values if not math.isnan(v)]
        result = list(subsetDataFrame[result_column])
        if variable_result in result:
            result = "NOK"
            array_nok.append(values)
            uuids_complete.append(uuid)
            if (len(values) > 0):
                candidates[uuid] = [result, values]
        else:
            result = "OK"
            array_ok.append(values)
            uuids_complete.append(uuid)
            if (len(values) > 0):
                candidates[uuid] = [result, values]
    return candidates, array_ok, array_nok, uuids_complete


def pipeline(use_case, df, id, variable_result,results,result_column, variable_interest=None, interval=None):
    candidates, array_ok, array_nok, uuids_complete = sort_array_ok_nok(df, id, variable_result, variable_interest, result_column)
    candidate_vars = ["value"]
    candidate_vars = [x for x in candidate_vars if x != result_column]
    num_cols_all = []
    df_og = df
    df_reset = df
    result_column_og = result_column


    for c in candidate_vars:
        df = df_reset
        variable_interest = c
        if use_case == "manufacturing":
            df_newFeatures = df[[id, variable_interest]]
            df_newFeatures = df_newFeatures.dropna()
            y_var = df[[id, result_column_og]].groupby(id).agg('last').dropna().reset_index()
            y_var = y_var[y_var[id].isin(df_newFeatures[id].values)]
            y_var = y_var[result_column_og].to_numpy()
            df, num_cols = prepare_dataset(df, id, variable_interest)
            df[id] = df.index
            df = df.reset_index(drop=True)

        else:
            df_newFeatures = df.select_dtypes(include=['number'])
            df, num_cols = prepare_dataset(df, id, variable_interest)
            df = df.dropna(axis=1)
            y_var = df[result_column_og + "last"].to_numpy()

        result_column = result_column_og + "last"
        max_accuracy = 0

        if not interval:
            interval = [2, 5, 10]
        else:
            interval = interval
        max_i = interval[0]
        accuracy_baseline = 0
        max_features = []
        try:
            for i in interval:
                df_new = generate_interval_features(df, i, variable_interest + "list")
                df_new = df_new.dropna()
                var_interval = df_new.select_dtypes(include=np.number).columns.tolist()
                var_interval = [x for x in var_interval if x != id]
                accuracy, used_features, _ = tc.learn_tree(df_new, result_column, var_interval, variable_result, False)
                if accuracy > accuracy_baseline:
                    accuracy_baseline = accuracy
                    max_i = i
                    max_features = used_features
        except Exception as e:
            print("ERROR INTERVAL FEATURES", e)
            pass
        df = generate_interval_features(df, max_i, variable_interest + "list")

        if accuracy_baseline > max_accuracy:
            max_accuracy = accuracy_baseline
            var_interval = max_features
        else:
            var_interval = []
        print("Calculated interval-based features...")

        # pattern based

        # candidate_thresholds = get_distribution(array_ok, array_nok)
        # df, var_pattern = create_latent_variables(candidate_thresholds, df, variable_interest)  ##FIXME: INCLUDE PATTERN BASED
        # try:
        #    accuracy, var_pattern, _ = tc.learn_tree(df, result_column, var_pattern, variable_result)
        #    if accuracy > max_accuracy:
        #        for var in var_pattern:
        #            var_interval.append(var)
        #    print("Calculated pattern-based features...")
        # except Exception as e:
        #    print("ERROR PATTERN BASED FEATURES", e)
        #    pass

        # #global features
        # df_newFeatures = df_newFeatures.dropna().reset_index()
        # try:                                                                                                             ##FIXME: add global features again
        #     global_features = generate_global_features(df_newFeatures, y_var, id, variable_interest)
        #     df = pd.merge(df, global_features, on=id)
        # except Exception as e:
        #     print("ERROR GLOBAL FEATURES", e)
        #     pass

        to_drop = []
        for d in df.columns:
            if np.inf in df[d].values:
                to_drop.append(d)
        df = df.drop(columns=to_drop)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        accuracy, num_cols, _ = tc.learn_tree(df, result_column, num_cols, variable_result)

        print("Calculated global features...")

        # combined
        for v in var_interval:
            num_cols.append(v)
        num_cols = [x for x in num_cols if x != id]
        df_og = pd.merge(df, df_og, on=id, how="outer", suffixes=('', '_y'))
        num_cols_all.extend(num_cols)

    df_og.drop_duplicates("uuid", inplace=True)
    y_var_bool = []
    for y in df_og[result_column]:
        if y=="ok":
            y_var_bool.append(1)
        else:
            y_var_bool.append(0)
    df_corr = df_og
    df_corr["result"] = y_var_bool
    corr = df_corr.corr(numeric_only=True)["result"].abs()
    corr = corr[corr > 0.1]
    colum = corr.index
    df_onlyrel = df_corr[colum]
    df_onlyrel = df_onlyrel.dropna()
    df_onlyrel[result_column] = df_corr[result_column]
    df_corr = df_onlyrel
    df_corr.drop("result", axis=1, inplace=True)
    _, _, rules_ts = tc.learn_tree(df_corr, result_column, num_cols_all, variable_result, results, True)
    print("\n---------------------------------------------------------------------------------\nedt-ts and bdt combined")
    rules_digiemine = em.define(df_og, result_column, True)
    return rules_ts, rules_digiemine

def main(file, use_case, id, variable_result, results, result_column, variable_interest):

    df = pd.read_csv(file)
    df = df.rename(columns={"timestamp": "time:timestamp"})
    rules_ts, rules_digiemine= pipeline(use_case, df, id, variable_result, results, result_column, variable_interest)
    return rules_ts, rules_digiemine
