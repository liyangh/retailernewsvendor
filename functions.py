import sys, os
import numpy as np
import csv
import itertools

###############################################################################
        
def closest(lst, v_desired): 
    
    closest_v_ind = min(range(len(lst)), key = lambda i: abs(lst[i]-v_desired))
    
    return (closest_v_ind, lst[closest_v_ind])

###############################################################################
###############################################################################
    
def all_combo(lst_players):
    
    subcoalitions = []
    for L in range(1, len(lst_players)+1):
        for subset in itertools.combinations(lst_players, L):
            subcoalitions.append(subset)
    
    return subcoalitions
    
###############################################################################
###############################################################################

def csv_to_lists(DATA_DIR, BY_ROW_COL=0, START_ROW=1, START_COL=1, DATA_TP=1):
    # DATA_DIR: data file directory
    # BY_ROW_COL: '0' means to output by row, and '1' means by column
    # START_ROW: first row to output
    # START_Column: first column to output
    # DATA_TP: "0" means text, "1" means integer, "2" means float
    
    data_in = csv.reader(open(DATA_DIR, 'r'))
    data_out = []
    row_ct = 0
    
    if BY_ROW_COL == 0:
        for row in data_in:
            row_ct += 1
            if row_ct < START_ROW:
                continue
            row_out = row[(START_COL-1):]
            for i in range(len(row_out)):
                try:
                    if DATA_TP == 0:
                        row_out[i] = row_out[i]
                    if DATA_TP == 1:
                        row_out[i] = int(row_out[i])
                    if DATA_TP == 2:
                        row_out[i] = float(row_out[i])
                except ValueError:
                    row_out[i] = row_out[i]
                data_out.append(row_out)
    elif BY_ROW_COL == 1:
        for row in data_in:
            row_ct += 1
            if row_ct < START_ROW:
                continue
            row_out = row[(START_COL-1):]
            if row_ct == START_ROW:
                data_out = [ [] for i in range(len(row_out)) ]
            for i in range(len(row_out)):
                try:
                    if DATA_TP == 0:
                        row_out[i] = row_out[i]
                    if DATA_TP == 1:
                        row_out[i] = int(row_out[i])
                    if DATA_TP == 2:
                        row_out[i] = float(row_out[i])
                except ValueError:
                    row_out[i] = row_out[i]
                data_out[i].append(row_out[i])
    else:
        print("please specify correct data output method:")
        print("'0' means by row, and '1' means by column")
    
    return(data_out)

###############################################################################
###############################################################################

def csv_dict_writer(path, fieldnames, data):
    """
    Writes a CSV file using DictWriter
    """
    with open(path, "w", newline='') as out_file:
        writer = csv.DictWriter(
                out_file, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

###############################################################################
###############################################################################

def dict_key_switch(dict_in):
    # dict_in is a dictionary of lists
    # e.g. dict_in = {1:[2,3], 2:[5,6]}
    # we desire dict_out = {0:{1:2, 2:5}, 1:{1:3, 2:6}}
    
    key_1_lst = list(dict_in)
    key_2_lst = range(len(dict_in[key_1_lst[0]]))
    
    dict_out = {}
    for key_2 in key_2_lst:
        dict_out[key_2] = {}
        for key_1 in key_1_lst:
            dict_out[key_2][key_1] = dict_in[key_1][key_2]
    
    return dict_out

###############################################################################
###############################################################################

def kw_kwh_convert(value_or_list, timescale, convert_direction):
    # convert_direction: '1': from kW to kWh; '2': from kWh to kW
    # timescale in [min]
    
    if convert_direction != 1 and convert_direction != 2:
        print("convert_direction value for kw_kwh_convert has to be" 
              + " '1' or '2': '1': from kW to kWh; '2': from kWh to kW")
        return
    if type(value_or_list) == list:
        if convert_direction == 1:
            output = [x * timescale / 60 for x in value_or_list]
        else:
            output = [x * 60 / timescale for x in value_or_list]
    else:
        try:
            if convert_direction == 1:
                output = value_or_list * timescale / 60 
            else:
                output = value_or_list * 60 / timescale
        except TypeError:
            print("input for kw_kwh_convert must be a value or a list.")
            return
    
    return output