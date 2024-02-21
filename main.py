import random
import techdraw.main as digiedraw
import numpy as np
import regex as re
import tsalignment as tsa
import os
import pandas as pd
import csv
import yaml
import dateutil
import matplotlib.pyplot as plt
from datetime import datetime
import compressor
from edt import edt_main, bdt
from edtts import time_series as ts
import fitz
import json

def remove_quotes(string):
    string = string.replace("'", "")
    return string
def split_string(string):
    newString = string.split()
    finalValue = newString[1]
    return finalValue
def get_times(fp,data):
    timeValue = -1
    for cnt, line in enumerate(fp):
        if 'value:' in line:
            if line.strip() != 'value: active':
                timeValue = cnt + 1
                measureValue = float(split_string(line.strip()))
        if cnt == timeValue:
            measureTime = split_string(line.strip())
            measureTime = remove_quotes(measureTime)
            newValue = dateutil.parser.parse(measureTime)
            timestamp = datetime.timestamp(newValue)
            data['measures'].append({
                'value': measureValue,
                'timestamp': timestamp
            })
def get_times_new(fp,data):
    for cnt, line in enumerate(fp):
        if 'value' in line:
            if line.strip() != 'value: active':
                val =line.strip()
                l = val.split()
                try:
                    time = l[3].split("=>")[1][:-1]
                    time = time[1:-1]
                    measureValue = float(l[2].split("=>")[1][:-1])
                    newValue = dateutil.parser.parse(time)
                    timestamp = datetime.timestamp(newValue)
                    data['measures'].append({
                        'value': measureValue,
                        'timestamp': timestamp
                    })
                except:
                    pass
def get_td_tolerances(drawing):
    uuid = str(random.randint(1000, 9999))
    db = "localhost"
    eps = 1
    result = digiedraw.main(uuid,drawing,db,eps)
    td_tolerances = []
    result = json.loads(result)
    result = result["No details"]
    for k in result:
        td_tolerances.append(k)
    return result #otherwise return td_tolerances
def extract_boundaries(td_tolerances):
    boundary = []
    bound_dict = {}
    count = 1
    for tol in td_tolerances:
        tol_list = re.findall(r'\d+\.\d+', tol)
        if tol_list:
            maxi = max(tol_list)
            if len(tol_list) == 2:
                mini = min(tol_list)
                up = float(maxi) + float(mini)
                down = float(maxi) - float(mini)
                boundary.append([up, down])
                bound_dict[str(tol)]= {"tol": {"bound"+str(count): up, "bound"+str(count+1):down}, "dims": td_tolerances[tol]}

            elif len(tol_list) == 3:
                temp = []
                for t in tol_list:
                    if t != maxi:
                        if float(t) == 0:
                            temp.append(float(maxi))
                        else:
                            sign = tol.split(t)[0]
                            sign = sign[-2]
                            if "+" in sign:
                                val = float(maxi) + float(t)
                            elif "-" in sign:
                                val = float(maxi) - float(t)
                            temp.append(val)
                up = max(temp)
                down = min(temp)
                boundary.append([up, down])
                bound_dict[str(tol)]= {"tol": {"bound"+str(count): up, "bound"+str(count+1):down}, "dims": td_tolerances[tol]}
        count +=2
    boundary = np.around(boundary, 3)
    return boundary, bound_dict
def getRelevantFiles(path,val):
    get_super_uuid = "Spawn GV12 Production"
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
                        inst.append([sinst, tinst]) #erstes instanz vom gv 12 spawn productin process, zweites element von measuring prozess
    #print(inst)
    return name,inst
def data_preparation(file,path,type):
    if '.yaml' in file:
        openfile = path + file
        data = {}
        stream = open(openfile)
        docs = yaml.load_all(stream,Loader=yaml.Loader)
        for doc in docs:
            for k, v in doc.items():
                try:
                    if (k == 'log'):
                        instance = v['trace']['concept:name']
                    if (k == 'event'):
                        try:
                            qr = v['data']['data_values']['qr']
                        except KeyError:
                            qr = "dummy"
                            #pass
                except TypeError:
                    continue
        data['qr'] = qr
        data['instance'] = instance
        data['measures'] = []
        with open(openfile) as fp:
            if type == "new":
                get_times_new(fp,data)
            else:
                get_times(fp,data)
            data["measures"].sort(key=lambda d: d["timestamp"])
            firsTime = data["measures"][0]["timestamp"] * 1000
            for i in data["measures"]:
                cmd = i["timestamp"] * 1000
                otherInt = cmd - firsTime
                i["timestamp"] = int(otherInt)
            final = json.dumps(data,indent=4)
    return final
def get_ok_nok(file,path):
    status = "ok"
    if '.yaml' in file:
        try:
            openfile = path + file
        except:
            return
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
                            values = v['data']['data_values']['qc2']
                            for key, val in values.items():
                                for k1, v1 in val.items():
                                    #print(k1)
                                    if (k1=="Zylinderform" or k1=="Distanz Z" or k1=="Mitte Z" or k1=="Rechtwinkligkeit"): #nur durchmesser werte relevant
                                        continue
                                    for k2,v2 in v1.items():
                                        if (k2=="status"):
                                            if (v2 == "nok"):
                                                status = "nok"
                                                return status

                           # print(status)
                        except KeyError:
                            qr = "dummy"
                            # pass
                except TypeError:
                    continue

    #print(status)
    return status
def extreme_values(compress,boundaries,mode="extreme"):
    values = []
    segments = []
    seg, line, areas = tsa.find_sline(compress, boundaries, direction="value")
    #print(seg, line, areas)
    for a in range(0, len(areas), 2):
        segments.append(str(compress["timestamp"][areas[a]]))
        area = compress["value"][areas[a]:areas[a + 1] + 1].tolist()
        time = compress["timestamp"][areas[a]:areas[a + 1] + 1].tolist()
        dic = {}
        if len(area)>1:
            if mode == 'freq':
                #--------------------------------frequent
                for t in range(0,len(time)-1):
                    if area[t] < 30:
                        if area[t] in dic:
                            dic[area[t]] = dic[area[t]] + (time[t+1] - time[t])
                        else:
                            dic[area[t]] = time[t+1] - time[t]
                try:
                    fin_max = max(dic, key=dic.get)
                    values.append(fin_max)
                except:
                    pass
            else:
            #--------------------------------extreme
                #meanval = mean(area)
                #if meanval < line:
                extreme_min = min(area)
                #else:
                extreme_max = max(area)
                values.append(extreme_min)
                values.append(extreme_max)
        else:
            values.append(area[0])
    #print(values, segments)
    return values,segments
# def tolerance_analysis(boundaries,compress,ideal=None,mode='extreme'):
#     values,areas = extreme_values(compress,boundaries,mode)
#     #print(areas)
#     relevant={}
#     allsegs = {}
#     status = True
#     return values

def get_data_from_yaml_and_compress(name, use_case):
    file_measuring = name[1] + ".xes.yaml"
    data = data_preparation(file_measuring, path, "old")
    file_super = name[0] + ".xes.yaml"
    if use_case!="turm":
        status = get_ok_nok(file_super, path)
    else:
        status = "n/a"
    d = json.loads(data)
    instance = d['instance']
    #qr = d['qr']
    data = pd.DataFrame(d['measures'])
    newData = data
    newData["value"] = newData.value.astype(float)
    newData["timestamp"] = newData.timestamp.astype(int)
    #print("newData:", newData)
    compress = compressor.approach_5(newData)
    plt.plot(newData["timestamp"], newData["value"], color='b', label=f'original')
    # plt.scatter(compress["timestamp"],compress["value"],color="r")
    plt.plot(compress["timestamp"], compress["value"], color="r", label=f'compressed')
    plt.legend()
    #plt.show()
    newData = newData[newData.value != 999.99]
    newData = newData.reset_index(drop=True)
    df = pd.DataFrame(newData, columns=["value","timestamp"])
    df["status"] = status
    df["uuid"] = instance
    return instance, compress, status, df

def get_infos(file, file_ts, use_case):
    # start digiedraw, get boundaries
    if use_case=="turm":
        boundaries = [[22.1, 21.9], [17.8, 17.2], [16.1, 15.9]]
    else:
        td_tolerances = get_td_tolerances(drawing=drawing)
        boundaries, bound_dict = extract_boundaries(td_tolerances)
        #repl = {"'": "\""}
        #bound_dict = {key: repl.get(value, value) for key, value in bound_dict.items()}
        print(bound_dict)
        with open(use_case+"tolerances_extracted.txt", "w") as outfile:
            json.dump(bound_dict, outfile)
    #print(boundaries)
    #get names of relevant log files
    names, inst = getRelevantFiles(path, "Measuring") # get all relevant instances from index.txt, return yaml file names und instance names
    #status_total = []
    df_timeseries = pd.DataFrame(columns=["value", "timestamp", "status", "uuid"])
    qa_turm_batch11 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]

    with open("archive/measuring_results.csv", "w") as f:
        header = ["uuid", "status", "seg1", "seg2", "seg3", "seg4", "seg5"]
        writer = csv.writer(f)
        writer.writerow(header)
        count = 0
        for name in names:
             data = []
             uuid, compressed_data, status, df_ts = get_data_from_yaml_and_compress(name, use_case) #get data from yaml files and compress them using approach 5 for each yaml log file
             if use_case=="turm":
                status = qa_turm_batch11[count]
             df_timeseries = df_timeseries.append(df_ts)

             #status_total.append((status))
             #print(pd.Series(status_total).value_counts())
             data.append(uuid)
             data.append(status)
             values,_ =  extreme_values(compressed_data,boundaries)
             for v in values[:4]:
                 data.append(v)
             count += 1
             writer.writerow(data)
        #    #print(relevant, status)
        #print(df_timeseries)
    df = pd.read_csv("archive/measuring_results.csv")
    add_colums = ["bound1", "bound2", "bound3", "bound4", "bound5", "bound6", "bound7", "bound8", "bound9", "bound10", "bound11", "bound12"]
    boundaries = [item for sublist in boundaries for item in sublist]
    count = 0
    for col in add_colums:
        df[col] = boundaries[count]
        df_timeseries[col] = boundaries[count]
        count +=1
    df_timeseries.to_csv(file_ts, index=False)
    df.to_csv(file, index=False)

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

def visualize_on_drawing(drawing, file_tol, str_rule):
    #get rule
    #map variable in rule to box in drawing
    result = load_json_from_file(file_tol)
    #used_boundaries = ["15.15 + 0.05 0.00", '1.5 + 0.1 0.0']
#     str_rule = "|--- row.bound2<row.int6valuelist1_max <= 0.50 \
# |   |--- row.bound7<row.int3valuelist1_min <= 0.50\
# |   |   |--- class: nok\
# |   |--- row.bound7<row.int3valuelist1_min >  0.50\
# |   |   |--- class: ok\
# |--- row.bound2<row.int6valuelist1_max >  0.50\
# |   |--- row.bound10<row.int11valuelist8_min <= 0.50\
# |   |   |--- row.bound1<=row.int6valuelist3_max <= 0.50\
# |   |   |   |--- class: nok\
# |   |   |--- row.bound1<=row.int6valuelist3_max >  0.50\
# |   |   |   |--- class: ok\
# |   |--- row.bound10<row.int11valuelist8_min >  0.50\
# |   |   |--- row.bound2>=row.int6valuelist2_max <= 0.50\
# |   |   |   |--- class: nok\
# |   |   |--- row.bound2>=row.int6valuelist2_max >  0.50\
# |   |   |   |--- class: nok"
#     str_rule2= "|--- int3valuelist1_max <= 19.18\
# |   |--- class: ok\
# |--- int3valuelist1_max >  4.3\
# |   |--- class: nok"
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
    doc.save(file_tol+"drawing.pdf")


def decision_mining(file, file_ts, use_case):
    result_column = "status"
    #print("\n-------------------------------------------------------\nfirst edt main with nataliia segments")
    #edt_main.define(file, result_column)
    #print("\n-------------------------------------------------------\nthen basic decision tree with nataliia segments")
    #bdt.run(file, result_column)
    print("\n-------------------------------------------------------\nthen edt-ts without nataliia segments")
    rulets, ruledem = ts.main(file_ts)
    return ruledem

if __name__ == '__main__':
    use_case = "synthetic"
    import sys
    sys.stdout = open(use_case + "_output.txt", "w")

    if use_case == "gv12":
        drawing = "/home/beatescheibel/PycharmProjects/digiemine_working/keyence/techdraw/drawings/GV_12.PDF"
        folder = "batch14"
        #drawing = "/home/beatescheibel/PycharmProjects/digiemine_working/keyence/techdraw/drawings/GV_12.PDF"
        #folder = "batch15"
        file = "measuring_results_plus_boundariesGV12.csv"
        file_ts = "measuring_intimeseriesGV12.csv"
        file_tol = "archive/gv12tolerances_extracted.txt"

    if use_case == "turm":
        drawing = "/home/beatescheibel/PycharmProjects/digiemine_working/keyence/techdraw/drawings/Turm.pdf"
        boundaries = [[22.1, 21.9], [17.8, 17.2], [16.1, 15.9]]
        qa = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        folder = "batch11"
        file = "measuring_results_plus_boundariesTurm.csv"
        file_ts = "measuring_intimeseries_turm.csv"


    if use_case == "synthetic":
        drawing = "/home/beatescheibel/PycharmProjects/digiemine_working/keyence/techdraw/drawings/Turm.pdf"
        boundaries = [[22.05, 21.9], [17.85, 17.15], [16.05, 15.95]]
        file_ts = "measuring_intimeseries_running2.csv"
        file = "measuring_results_plus_boundaries_running2.csv"
        folder = "running"

    name = re.findall("\/(\w*\.\w*)", drawing)
    path = '/home/beatescheibel/PycharmProjects/digiemine_working/keyence/timesequence/' + folder + '/'
    #get_infos(file, file_ts, use_case)
    rule = decision_mining(file, file_ts, use_case)
    visualize_on_drawing(drawing, file_tol, rule)
    sys.stdout.close()

