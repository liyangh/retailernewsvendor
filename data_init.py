import sys, os, errno
import numpy as np
import csv
#import itertools
from scipy.stats import norm, truncnorm
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

from functions import csv_to_lists, csv_dict_writer, kw_kwh_convert

#==============================================================================
#   Read load data
#==============================================================================

def load_input(N, NO_IN, RAND_IN, CUST_SEL, SIM_DT, START_T, SIM_T, L_DT, 
               GENERIC_L_EXP, CASE_OUT_DIR, CASE_OUT_NAME, DATA_FOLD, L_FOLD, 
               SEL_FOLD):
    
    # NO_IN: number of input datasets in the "Data_Inputs" folder,
    # RAND_IN: whether to randomly select modeled players from input datasets
    # CUST_SEL: whether to customize load profile selection, overwrite RAND_IN
    # N: Number of players, SIM_DT in [min], 
    # START_T: simulation start time, SIM_T: simulation length [days],
    # L_DT: load input data
    
###############################################################################
#   Individual Net Loads 

    # define load prediction dictionary load_prdct_dic that has N entries, 
    # and each entry is a list of load precition [kWh] of a player by timestep
    
    current_dir = os.path.dirname(__file__)
    cust_file = 'N%d_input_prof_ind.csv' % N
    l_file = 'Load_profile_%d.csv' 
    
    players = list(range(1, N+1))
    
    # if customized load profile selections are required, overwrite load data selection unless data not found
    custom_p_index = []
    if CUST_SEL == 1:
        mdl_p_dir = os.path.join(current_dir, DATA_FOLD, SEL_FOLD, cust_file)
        try:
            mdl_p_list = csv_to_lists(mdl_p_dir, 1, 2, 2)
        except:
            pass
        try:
            custom_p_index = mdl_p_list[0]
        except:
            custom_p_index = []
        if len(custom_p_index) != N:
            custom_p_index = []
                    
    if custom_p_index != []:
        mdl_p_ind = custom_p_index
    elif RAND_IN == 1:
        all_input_ind = range(1, NO_IN+1)
        mdl_p_ind = list(np.random.choice(all_input_ind, len(players)))
    else:
        mdl_p_ind = players
    
    end_t = START_T + timedelta(days = SIM_T)
    print(end_t)
    if end_t.year > START_T.year:
        cross_year = 1
        mo_list = list(range(START_T.month, 13))
        mo_list.extend(list(range(1, end_t.month+1)))
    else:
        cross_year = 0
        mo_list = list(range(START_T.month, end_t.month+1))
    tot_tsteps = int(SIM_T * 24 * 60 / SIM_DT)
    load_prdct_dic = {}
    
    for i in players:
        load_profile = []
        signal = 0
        data_end_time = 'N/A'
        print('Player %d:' % i)
        
        # import load profiles in [kW]
        for mo_n in range(len(mo_list)):
            load_raw = csv.reader(
                    open(os.path.join(current_dir, DATA_FOLD, 
                                      L_FOLD % (L_DT, mo_list[mo_n]), 
                                      l_file % mdl_p_ind[i-1]), 'r'))
            if mo_n == 0:
                for row_n, row in enumerate(load_raw):
                    if row_n == 0:
                        continue
                    entry_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    # bypass all the entries before entry_time reaches START_T
                    entry_time = entry_time.replace(year=START_T.year)
                    if entry_time < START_T:
                        continue
                    elif entry_time >= end_t:
                        print('input data end time: %s' % data_end_time)
                        break
                    if signal == 0:
                        print('input data start time: %s' % entry_time)
                        signal += 1
                    try:
                        l_value = abs(float(row[1]))
                    except ValueError:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    if l_value == 0.:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    load_profile.append(l_value)
                    data_end_time = entry_time
                
            elif mo_n == len(mo_list)-1:
                for row_n, row in enumerate(load_raw):
                    if row_n == 0:
                        continue
                    entry_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    if cross_year == 1:
                        entry_time = entry_time.replace(year=end_t.year)
                    else:
                        entry_time = entry_time.replace(year=START_T.year)
                    # bypass all the entries before entry_time reaches START_T
                    if entry_time >= end_t:
                        print('input data end time: %s' % data_end_time)
                        break
                    try:
                        l_value = abs(float(row[1]))
                    except ValueError:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    if l_value == 0.:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    load_profile.append(l_value)
                    data_end_time = entry_time
                
            else:
                for row_n, row in enumerate(load_raw):
                    if row_n == 0:
                        continue
                    entry_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    entry_time = entry_time.replace(year=START_T.year)
                    if cross_year == 1 and entry_time < START_T:
                        entry_time = entry_time.replace(year=end_t.year)
                    try:
                        l_value = abs(float(row[1]))
                    except ValueError:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    if l_value == 0.:
                        l_value = kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)
                    load_profile.append(l_value)
                    data_end_time = entry_time
                
        if len(load_profile) < tot_tsteps:
            load_profile = \
                [kw_kwh_convert(GENERIC_L_EXP, SIM_DT, 1)] * tot_tsteps
            
        load_sim = np.mean(np.array(load_profile).reshape(-1, \
                           int(SIM_DT/L_DT)), axis = 1) * float(SIM_DT/60)
        # load_sim is now an array in [kWh] for every timestep
        
        print('there are %d timesteps' % len(load_sim))
        load_prdct_dic[i] = list(load_sim)
    
    l_ind_print_list = []
    player_fldnames = ['Player', 'Profile Selection']
    
    for i in range(1, 1 + N):
        l_index_run_dict = {}
        l_index_run_dict['Player'] = i
        l_index_run_dict['Profile Selection'] = mdl_p_ind[i-1]
        l_ind_print_list.append(l_index_run_dict)
    
    csv_dict_writer(os.path.join(CASE_OUT_DIR, cust_file), 
                    player_fldnames, l_ind_print_list)

###############################################################################
#   Results
    
    # l_mdl_ind: modeled load profile index
    return load_prdct_dic

#==============================================================================
#   Coalition Value Calculation
#==============================================================================

def data_initial(NO_P, NO_IN_DAT, RANDOM_IN, CUSTOM_SEL, SIMLTN_DT, SIMLTN_T, 
                 SIM_START_T, DAT_DT, CASE_OUTPUT_DIR, CASE_OUTPUT_NAME, 
                 DAT_FOLDER, L_FOLDER, SEL_FOLDER, L_TYPE, RAND_LO, 
                 RAND_HI, GR_L_EXP, SCHD_L_FRAC_RANGE, L_ACTL_RANGE, 
                 SCHD_L_D_TO_MEAN, NONSCHD_L_D_TO_MEAN, IND_L_OUT, 
                 CASE_L_PDF_OUT, SNAP_T):
    # T in [days], 
    
    t_steps = int(SIMLTN_T * 24 * 60 / SIMLTN_DT)
    time_print = SIM_START_T
    time_list = []
    t_print_lst = []
    for t in range(t_steps):
        time_list.append(time_print)
        t_print_lst.append(time_print.strftime("%m-%d-%H"))
        time_print += timedelta(minutes = SIMLTN_DT)
    
    # All loads in [kWh]
    indiv_l_std_d_dic = {}
    schd_load_frac_dic = {}
    schd_l_exp_dic = {}
    schd_l_min_dic = {}
    schd_l_max_dic = {}
    schd_l_std_d_dic = {}
    schd_l_dist_dic = {}
    schd_l_trun_dist_dic = {}
    schd_l_trun_actl_dic = {}
    nonschd_l_exp_dic = {}
    nonschd_l_min_dic = {}
    nonschd_l_max_dic = {}
    nonschd_l_std_d_dic = {}
    nonschd_l_dist_dic = {}
    indiv_l_dist_dic = {}
    indiv_l_05perc = {}
    indiv_l_95perc = {}
    
    schd_l_trun_actl_sum_t = [0] * t_steps
    tot_l_exp_wo_data_t = [0] * t_steps
    tot_l_var_t = [0] * t_steps
    nonschd_l_exp_sum_t = [0] * t_steps
    nonschd_l_var_sum_t = [0] * t_steps
    
    if L_TYPE == 1:
        indiv_l_exp_dic = load_input(
            NO_P, NO_IN_DAT, RANDOM_IN, CUSTOM_SEL, SIMLTN_DT, SIM_START_T, 
            SIMLTN_T, DAT_DT, GR_L_EXP, CASE_OUTPUT_DIR, CASE_OUTPUT_NAME, 
            DAT_FOLDER, L_FOLDER, SEL_FOLDER)
        tot_l_exp_wo_data_t = \
            [sum(l_t) for l_t in zip(*list(indiv_l_exp_dic.values()))]
        if tot_l_exp_wo_data_t == []:
            tot_l_exp_wo_data_t = [0] * t_steps
    else:
        indiv_l_exp_dic = {}
        
    for i in range(1, NO_P+1):
        if L_TYPE == 0:
            indiv_l_exp_dic[i] = []
        
        indiv_l_std_d_dic[i] = []
        schd_l_exp_dic[i] = []
        schd_l_min_dic[i] = []
        schd_l_max_dic[i] = []
        schd_l_std_d_dic[i] = []
        schd_l_dist_dic[i] = []
        schd_l_trun_dist_dic[i] = []
        schd_l_trun_actl_dic[i] = []
        nonschd_l_exp_dic[i] = []
        nonschd_l_min_dic[i] = []
        nonschd_l_max_dic[i] = []
        nonschd_l_std_d_dic[i] = []
        nonschd_l_dist_dic[i] = []
        indiv_l_dist_dic[i] = []
        indiv_l_05perc[i] = []
        indiv_l_95perc[i] = []
        
        schd_load_frac_dic[i] = SCHD_L_FRAC_RANGE[0] \
            + (SCHD_L_FRAC_RANGE[1] - SCHD_L_FRAC_RANGE[0]) \
                * random.randint(RAND_LO, RAND_HI) / RAND_HI
        
        for t in range(t_steps):
            if L_TYPE == 0:
                indiv_l_exp_dic[i].append(
                    random.randint(RAND_LO, RAND_HI) / RAND_HI \
                        * GR_L_EXP * float(SIMLTN_DT/60))
                # individual load is now an array in [kWh] for every timestep
                tot_l_exp_wo_data_t[t] += indiv_l_exp_dic[i][t]
            schd_l_exp_dic[i].append(
                indiv_l_exp_dic[i][t] * schd_load_frac_dic[i])
            schd_l_min_dic[i].append(schd_l_exp_dic[i][t] * L_ACTL_RANGE[0])
            schd_l_max_dic[i].append(schd_l_exp_dic[i][t] * L_ACTL_RANGE[1])
            schd_l_std_d_dic[i].append(
                abs(schd_l_exp_dic[i][t] * SCHD_L_D_TO_MEAN))
            schd_l_dist_dic[i].append(
                norm(loc = schd_l_exp_dic[i][t], 
                     scale = schd_l_std_d_dic[i][t]))
            schd_l_trun_dist_dic[i].append(truncnorm(
                (schd_l_min_dic[i][t]-schd_l_exp_dic[i][t]) \
                    / schd_l_std_d_dic[i][t],
                (schd_l_max_dic[i][t]-schd_l_exp_dic[i][t]) \
                    / schd_l_std_d_dic[i][t],
                loc = schd_l_exp_dic[i][t], scale = schd_l_std_d_dic[i][t]))
            schd_l_trun_actl_dic[i].append(
                schd_l_trun_dist_dic[i][t].rvs(1)[0])
            schd_l_trun_actl_sum_t[t] += schd_l_trun_actl_dic[i][t]
            nonschd_l_exp_dic[i].append(
                indiv_l_exp_dic[i][t] - schd_l_exp_dic[i][t])
            nonschd_l_exp_sum_t[t] += nonschd_l_exp_dic[i][t]
            
            nonschd_l_min_dic[i].append(
                nonschd_l_exp_dic[i][t] * L_ACTL_RANGE[0])
            nonschd_l_max_dic[i].append(
                nonschd_l_exp_dic[i][t] * L_ACTL_RANGE[1])
            nonschd_l_std_d_dic[i].append(
                nonschd_l_exp_dic[i][t] * NONSCHD_L_D_TO_MEAN)
            nonschd_l_var_sum_t[t] += nonschd_l_std_d_dic[i][t] ** 2
            nonschd_l_dist_dic[i].append(
                norm(loc = nonschd_l_exp_dic[i][t], 
                     scale = nonschd_l_std_d_dic[i][t]))
            indiv_l_std_d_dic[i].append(
                (schd_l_std_d_dic[i][t]**2 + nonschd_l_std_d_dic[i][t]**2) \
                    ** (1/2))
            tot_l_var_t[t] += indiv_l_std_d_dic[i][t] ** 2
            indiv_l_dist_dic[i].append(
                norm(loc = indiv_l_exp_dic[i][t], 
                     scale = indiv_l_std_d_dic[i][t]))
            indiv_l_05perc[i].append(indiv_l_dist_dic[i][t].ppf(0.05))
            indiv_l_95perc[i].append(indiv_l_dist_dic[i][t].ppf(0.95))
        
        if IND_L_OUT == 1:
            indiv_l_fig = plt.figure(figsize=(7,4))
            indiv_l_fig.patch.set_alpha(0.0)
            plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.9, 
                                wspace=.1, hspace=.25)
            indiv_l = indiv_l_fig.add_subplot(111)
            
            indiv_l.plot(
                time_list, kw_kwh_convert(indiv_l_exp_dic[i], SIMLTN_DT, 2), 
                'k-', lw=1.5, label='load expectation')
            indiv_l.plot(
                time_list, kw_kwh_convert(indiv_l_05perc[i], SIMLTN_DT, 2),
                'b-.', lw=1, label='0.05 fractile')
            indiv_l.plot(
                time_list, kw_kwh_convert(indiv_l_95perc[i], SIMLTN_DT, 2),
                'r--', lw=1, label='0.95 fractile')
            indiv_l.stackplot(
                time_list, 
                kw_kwh_convert(indiv_l_05perc[i], SIMLTN_DT, 2),
                kw_kwh_convert(
                    np.subtract(indiv_l_exp_dic[i], indiv_l_05perc[i]), 
                    SIMLTN_DT, 2),
                kw_kwh_convert(
                    np.subtract(indiv_l_95perc[i], indiv_l_exp_dic[i]),
                    SIMLTN_DT, 2),
                colors=['None', 'lightseagreen', 'salmon'], alpha=0.5,
                edgecolors='None')
            
            indiv_l.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
            indiv_l.set_xlabel('Time (Month-Day-Hour)', fontsize=13)
            indiv_l.set_ylabel('Individual load (kW)', fontsize=13)
            indiv_l.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), 
                           ncol=3, fontsize=10)
            try:
                os.makedirs(os.path.join(CASE_OUTPUT_DIR, 'indiv_loads'))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            indiv_l_fig.savefig(
                os.path.join(CASE_OUTPUT_DIR, 'indiv_loads/p_%d_load.pdf' % i),
                format = 'pdf', dpi = 1000)
    
    tot_l_exp_w_data_t = [x + y for (x,y) in 
                           zip(nonschd_l_exp_sum_t, schd_l_trun_actl_sum_t)]
    tot_l_wo_data_dist_t = []
    tot_l_w_data_dist_t = []
    for t in range(t_steps):
        tot_l_wo_data_dist_t.append(
            norm(loc = tot_l_exp_wo_data_t[t], 
                 scale = tot_l_var_t[t] ** (1/2)))
        tot_l_w_data_dist_t.append(
            norm(loc = tot_l_exp_w_data_t[t], 
                 scale = nonschd_l_var_sum_t[t] ** (1/2)))
        
    tot_l_wo_data_05perc = [dist.ppf(.05) for dist in tot_l_wo_data_dist_t]
    tot_l_wo_data_95perc = [dist.ppf(.95) for dist in tot_l_wo_data_dist_t]
    tot_l_w_data_05perc = [dist.ppf(.05) for dist in tot_l_w_data_dist_t]
    tot_l_w_data_95perc = [dist.ppf(.95) for dist in tot_l_w_data_dist_t]
    
    if CASE_L_PDF_OUT == 1:
        tot_l_fig = plt.figure(figsize=(7,4))
        tot_l_fig.patch.set_alpha(0.0)
        plt.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.86, 
                            wspace=.1, hspace=.25)
        tot_l = tot_l_fig.add_subplot(111)
        tot_l.plot(
            time_list, kw_kwh_convert(tot_l_exp_wo_data_t, SIMLTN_DT, 2),
            'r-', lw=1.2, label='load exp. w/o data', alpha = 0.7)
        tot_l.plot(
            time_list, kw_kwh_convert(tot_l_wo_data_05perc, SIMLTN_DT, 2),
            c='saddlebrown', linestyle='--', lw=1, 
            label='0.05-0.95 fractile w/o data')
        
        tot_l.plot(
            time_list, kw_kwh_convert(tot_l_wo_data_95perc, SIMLTN_DT, 2),
            c='saddlebrown', linestyle='--', lw=1)
        #y_min, y_max = tot_l.get_ylim()
        #tot_l.vlines(time_list[SNAP_T], y_min, y_max, linestyle='-.', lw=1, 
                     #color='k', label='timestep for snapshot profit figure')
        tot_l.plot(
            time_list, kw_kwh_convert(tot_l_exp_w_data_t, SIMLTN_DT, 2),
            'b-', lw=1.5, label='load exp. w/ data', alpha = 0.7)
        #tot_l.plot(time_list, tot_l_w_data_05perc,
                   #'b-.', lw=1, label='0.05 fractile w/ data')
        #tot_l.plot(time_list, tot_l_w_data_95perc,
                   #'b--', lw=1, label='0.95 fractile w/ data')
        tot_l.stackplot(
            time_list, kw_kwh_convert(tot_l_w_data_05perc, SIMLTN_DT, 2),
            kw_kwh_convert(
                np.subtract(tot_l_w_data_95perc, tot_l_w_data_05perc),
                SIMLTN_DT, 2),
            colors=['None', 'darkcyan'], edgecolors='None', alpha=0.5,
            labels=[None,'0.05-0.95 fractile w/ data'])
        
        
        tot_l.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        tot_l.set_xlabel('Hour of Day', fontsize=13)
        tot_l.set_ylabel('Total load (kW)', fontsize=13)
        
        tot_l.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), 
                     ncol=2, fontsize=10)
        tot_l_fig.savefig(
                os.path.join(CASE_OUTPUT_DIR, 'total_load.pdf'),
                format = 'pdf', dpi = 1000)
    
                
###############################################################################
    #   Results
    
    # profit in ¢
    # All load output in [kWh]
    
    return (indiv_l_exp_dic, schd_l_exp_dic, schd_load_frac_dic,
            schd_l_min_dic, schd_l_max_dic, schd_l_std_d_dic, 
            schd_l_dist_dic, schd_l_trun_actl_dic,
            nonschd_l_exp_dic, nonschd_l_min_dic, nonschd_l_max_dic,
            nonschd_l_std_d_dic, nonschd_l_dist_dic, schd_l_trun_actl_sum_t, 
            nonschd_l_exp_sum_t, tot_l_exp_wo_data_t, tot_l_exp_w_data_t, 
            tot_l_wo_data_dist_t, tot_l_w_data_dist_t, time_list)