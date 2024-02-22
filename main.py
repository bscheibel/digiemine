# Copyright (C) 2024  Beate Wais
# This file is part of digiemine.
#
# digiemine is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# digiemine is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# digiemine.  If not, see <http://www.gnu.org/licenses/>.


import random
import techdraw.main as digiedraw
import numpy as np
import regex as re
import os
import pandas as pd
import yaml
import dateutil
from datetime import datetime
from edt_ts import time_series as ts
import fitz
import json

def get_times(fp,data):
    timeValue = -1
    for cnt, line in enumerate(fp):
        if 'value:' in line:
            if line.strip() != 'value: active':
                timeValue = cnt + 1
                measureValue = float(line.strip().split()[1])
        if cnt == timeValue:
            timev = line.strip().split()[1].replace("'", "")
            newValue = dateutil.parser.parse(timev)
            timestamp = datetime.timestamp(newValue)
            data['measures'].append({
                'value': measureValue,
                'timestamp': timestamp
            })

def get_tolerances(drawing):
    uuid = str(random.randint(1000, 9999))
    db = "localhost"
    eps = 1
    result = digiedraw.main(uuid,drawing,db,eps)
    result = json.loads(result)
    result = result["No details"]
    boundary = []
    bound_dict = {}
    count = 1
    for tol in result:
        tol_list = re.findall(r'\d+\.\d+', tol)
        if tol_list:
            maximum = max(tol_list)
            if len(tol_list) == 2:
                minimum = min(tol_list)
                up = float(maximum) + float(minimum)
                down = float(maximum) - float(minimum)
                boundary.append([up, down])
                bound_dict[str(tol)]= {"tol": {"boundary"+str(count): up, "boundary"+str(count+1):down}, "dims": result[tol]}
            elif len(tol_list) == 3:
                temp = []
                for t in tol_list:
                    if t != maximum:
                        if float(t) == 0:
                            temp.append(float(maximum))
                        else:
                            sign = tol.split(t)[0]
                            sign = sign[-2]
                            if "+" in sign:
                                val = float(maximum) + float(t)
                            elif "-" in sign:
                                val = float(maximum) - float(t)
                            temp.append(val)
                up = max(temp)
                down = min(temp)
                boundary.append([up, down])
                bound_dict[str(tol)]= {"tol": {"boundary"+str(count): up, "boundary"+str(count+1):down}, "dims": result[tol]}
        count +=2
    boundary = np.around(boundary, 3)
    return boundary, bound_dict

def get_file_names(path,val, use_case):
    if use_case=="gv12":
        get_super_uuid = "Spawn GV12 Production"
    if use_case=="turm":
        get_super_uuid = "Spawn Turm Production"
    name = []
    inst = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                filename = path + file
                file1 = open(filename, 'r')
                Lines = file1.readlines()
                count = 0
                for line in Lines:
                    count += 1
                    string = line.strip()
                    if get_super_uuid in string:
                        sname =  re.search(r"\((.*?)\)", string).group(1)
                        sinst = string.split("-",-1)[-1]
                    if val in string:
                        tname = re.search(r"\((.*?)\)", string).group(1)
                        tinst = string.split("-",-1)[-1]
                        name.append([sname,tname])
                        inst.append([sinst, tinst])
    return name,inst

def get_status(file,path):
    status = "ok"
    if '.yaml' in file:
        try:
            openfile = path + file
        except:
            return
        stream = open(openfile)
        docs = yaml.load_all(stream, Loader=yaml.Loader)
        for doc in docs:
            for k, v in doc.items():
                try:
                    if (k == 'event'):
                        try:
                            values = v['data']['data_values']['qc2']
                            for key, val in values.items():
                                for k1, v1 in val.items():
                                    if (k1=="Zylinderform" or k1=="Distanz Z" or k1=="Mitte Z" or k1=="Rechtwinkligkeit"): #nur durchmesser werte relevant
                                        continue
                                    for k2,v2 in v1.items():
                                        if (k2=="status"):
                                            if (v2 == "nok"):
                                                status = "nok"
                                                return status
                        except KeyError:
                            continue
                except TypeError:
                    continue
    return status

def get_infos_from_logs(name, use_case):
    file = name[1] + ".xes.yaml"
    if '.yaml' in file:
        openfile = path + file
        data = {}
        stream = open(openfile)
        docs = yaml.load_all(stream, Loader=yaml.Loader)
        for doc in docs:
            for k, v in doc.items():
                try:
                    if (k == 'log'):
                        instance = v['trace']['concept:name']
                    if (k == 'event'):
                        try:
                            qr = v['data']['data_values']['qr']
                        except KeyError:
                            qr = "na"
                except TypeError:
                    continue
        data['qr'] = qr
        data['instance'] = instance
        data['measures'] = []
        with open(openfile) as fp:
            get_times(fp, data)
            data["measures"].sort(key=lambda d: d["timestamp"])
            firsTime = data["measures"][0]["timestamp"] * 1000
            for i in data["measures"]:
                cmd = i["timestamp"] * 1000
                otherInt = cmd - firsTime
                i["timestamp"] = int(otherInt)
            data = json.dumps(data, indent=4)
    file_super = name[0] + ".xes.yaml"
    if use_case!="turm":
        status = get_status(file_super, path)
    else:
        status = "n/a"
    d = json.loads(data)
    instance = d['instance']
    data = pd.DataFrame(d['measures'])
    newData = data
    newData["value"] = newData.value.astype(float)
    newData["timestamp"] = newData.timestamp.astype(int)
    newData = newData[newData.value != 999.99]
    newData = newData.reset_index(drop=True)
    df = pd.DataFrame(newData, columns=["value","timestamp"])
    df["status"] = status
    df["uuid"] = instance
    return df

def get_infos(file_ts, use_case):
    if use_case=="turm":
        boundaries = [[22.1, 21.9], [17.8, 17.2], [16.1, 15.9]]
    else:
        boundaries, bound_dict = get_tolerances(drawing=drawing)
        with open("data/"+ use_case+"tolerances_extracted.txt", "w") as outfile:
            json.dump(bound_dict, outfile)
    names, inst = get_file_names(path, "Measuring", use_case)
    df_timeseries = pd.DataFrame(columns=["value", "timestamp", "status", "uuid"])

    with open("archive/measuring_results.csv", "w") as f:
        for name in names:
             df_ts = get_infos_from_logs(name, use_case)
             df_timeseries = df_timeseries.append(df_ts)
    if use_case == "gv12":
        add_colums = ["bound1", "bound2", "bound3", "bound4", "bound5", "bound6", "bound7", "bound8", "bound9", "bound10", "bound11", "bound12"]
    if use_case == "turm":
        add_colums = ["bound1", "bound2", "bound3", "bound4", "bound5", "bound6"]
    boundaries = [item for sublist in boundaries for item in sublist]
    count = 0
    for col in add_colums:
        df_timeseries[col] = boundaries[count]
        count +=1
    df_timeseries.to_csv(file_ts, index=False)

def draw_rectangle_on_pdf(array_coords,input_filename, output_filename):
    coords_x = float(array_coords[0])
    coords_y = float(array_coords[1])
    coords_xmax = float(array_coords[2])
    coords_ymax = float(array_coords[3])
    doc = fitz.open(input_filename)
    for page in doc:
        page.draw_rect([coords_x, coords_y, coords_xmax, coords_ymax], color=(1, 0, 0), width=2)
    doc.save(output_filename)

def load_json_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from '{filename}'.")
        return None

def visualize_on_drawing(drawing, file_tol, str_rule, use_case):
    result = load_json_from_file(file_tol)
    bounds = re.findall("(boundary\d*)", str_rule)
    bounds_approx = []
    if not bounds:
        bounds_approx = re.findall("(\W\d+\D?\d*)", str_rule)
    bounds_found = []
    bounds_approx = list(set(bounds_approx))
    df = pd.json_normalize(result, sep='_')
    df = df.to_dict(orient='records')[0]
    if bounds_approx:
        for ba in bounds_approx:
            for k in result:
                bounds = result[k]["tol"]
                keysList = list(bounds.keys())
                upper_k = keysList[0]
                lower_k = keysList[1]
                if float(ba) <= float(bounds[upper_k]) and float(ba) >= float(bounds[lower_k]):
                    bounds_found.append(lower_k)

            min_abstand = 100000
            min_bounds = ""
            for k,v in df.items():
                bounds = re.findall("(boundary\d*)", k)
                if bounds:
                    value = df[k]
                    abstand = abs(float(value)-float(ba))
                    if abstand < min_abstand:
                        min_abstand = abstand
                        min_bounds = bounds[0]
            print("Be careful! No exact boundaries were found! Output is only approximation!")
            bounds_found.append(min_bounds)

    bounds.extend(bounds_found)
    bounds = list(set(bounds))
    doc = fitz.open(drawing)
    for page in doc:
        for b in bounds:
            for k in result:
                    if b in result[k]["tol"]:
                        dims = result[k]["dims"]
                        bounds = result[k]["tol"]
                        coords_x = float(dims[0])
                        coords_y = float(dims[1])
                        coords_xmax = float(dims[2])
                        coords_ymax = float(dims[3])
                        page.draw_rect([coords_x, coords_y, coords_xmax, coords_ymax], color=(1, 0, 0), width=2)
    doc.save("results/" + use_case+"drawing.pdf")



if __name__ == '__main__':
    use_case = "synthetic"
    wd = "/home/beatescheibel/PycharmProjects/"

    if use_case == "gv12":
        drawing = wd + "digiemine/techdraw/drawings/GV_12.PDF"
        folder = "batch14"
        file_ts = "data/measuring_intimeseriesGV12.csv"
        file_tol = "data/gv12tolerances_extracted.txt"

    if use_case == "turm":
        drawing = wd + "/digiemine/techdraw/drawings/Turm.pdf"
        boundaries = [[22.1, 21.9], [17.8, 17.2], [16.1, 15.9]]
        qa = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        folder = "batch11"
        file_ts = "data/measuring_intimeseries_turm.csv"


    if use_case == "synthetic":
        drawing = wd + "digiemine/techdraw/drawings/Turm.pdf"
        boundaries = [[22.05, 21.9], [17.85, 17.15], [16.05, 15.95]]
        file_ts = "data/measuring_intimeseries_running.csv"
        folder = "running"

    name = re.findall("\/(\w*\.\w*)", drawing)
    path = wd + "digiemine/timesequence/" + folder + "/"
    #get_infos(file_ts, use_case)
    id = "uuid"
    results = ['ok', 'nok']
    result_column = 'status'
    variable_result = 'nok'

    variable_interest = "value"
    rulets, ruledem = ts.main(file_ts,  use_case, id, variable_result, results, result_column, variable_interest)
    # visualize_on_drawing(drawing, file_tol, ruledem, use_case)

