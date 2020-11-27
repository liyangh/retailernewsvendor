import sys, os, errno
#import csv
from datetime import datetime#, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import random
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
import time

from functions import closest, all_combo, dict_key_switch
from shapley_calc import shapley
from nucleolus_calc import nucleolus
from prof_calc import profit_calc, c_v_calc
from data_init import data_initial
from single_tstep import prof_snap
from imputation import coop_imp

mdl_start_t = time.time()

#==============================================================================
# Global Inputs
#==============================================================================

No_data_input = 34 # number of input datasets available in the "Data_Inputs" folder
load_profile_dt = 15 # [min]

simulation_start_time = datetime(2013,1,1,0,0,0) #datetime(year,month,day,hr,min,sec)

simulation_dt = 15 # [min]
simulation_length = 1 # [day]

# electricity prices in £/kWh
price_retail = [0.10] * 96 # date, link
price_wholesale = [0.06] * 96 # procurement price c, date, link
price_pos_imb = [0.02] * 96 # imbalance price for surplus, additive inverse of holding price h
price_neg_imb = [0.16] * 96 # imbalance price for shortage, stock-out price v


#==============================================================================
# Case Inputs
#==============================================================================

No_cases = 1 # number of cases analyzed

No_players = [8, 20, 30] # each entry represents a case, if No_cases < size(No_players), only No_cases will be analyzed
rand_p_sel = 1 # "1" means to randomely select the players from all the avialble datasets
custom_p_sel = 1 # user input file, HIGHEST overwrite priority

generic_l_exp = [3, 3, 3] # in [kW] integers! each entry represents a case, if No_cases < size(No_PV), only No_cases will be analyzed
range_l_to_mean = [[0.,2.], [0.,2.], [0.,2.]] # applied for the truncated 
range_schd_load_frac = [[0.1,0.9], [0.1,0.5], [0.1,0.5]] # each entry represents a case, if No_cases < size(battery_count), only No_cases will be analyzed
schd_l_d_to_mean = [1., 1., 1.]
nonschd_l_d_to_mean = [0.5, 0.5, 0.5]

uni_rand_floor = 1 # lowest value to generate a random integer number in a uniform distribution
uni_rand_ceil = 10 # highest value to generate a random integer number in a uniform distribution

No_quan_wholesale = 100 # Used when single_t_out = 1: number of tested wholesale imports, evenly distributed over the possible import quantities
No_reals = 10**3 # Number of realizations of pdfs under each wholesale import quantity

#==============================================================================
# Computation Inputs
#==============================================================================

closest_bid_on = 0 # whether to use the closest bid that is computed based on No_quan_wholesale
l_input_tp = 1 # type of load inputs used: '0' generic loads; '1': load data
bid_type = [1, 1, 1] # type of bid used in deciding the wholesale quantity: '0' expectation; '1': cost ratio
sc_v_def = 0 # coalition value definition: '0': use distribution with grand coalition data for all subcoalition value calc; '1': use distribution with only corresponding subcoalition data 

#==============================================================================
# Output Configuration
#==============================================================================

indiv_l_out = 1 # whether to output individual player loads
case_l_pdf_out = 1 # whether to output the total load pdf for each case
single_t_out = 1 # whether to output the profit distribution snapshot of a single timestep

#==============================================================================
# Output Directory Inputs
#==============================================================================

# data input folders
data_in_folder = 'data_inputs'
load_dat_folder = 'by_month/profiles_TC5_%dmin_month%d'
cust_sel_folder = 'selections'
# data output folders
results_folder = 'results'
res_sum_folder = 'summary'
res_case_folder = 'N%d_T%d_%s_schd%d-%d'
res_case_snapshot_folder = 'single_timestep'
res_case_imp_folder = 'imputations'

#==============================================================================
# Model Pre-run Tests
#==============================================================================

for case in range(No_cases):

    if range_schd_load_frac[case][0] == 0.:
        print("minimum range_schd_load_frac needs to be greater than 0")
        sys.exit()

#==============================================================================
# Input Data Processing
#==============================================================================

snap_tstep = int(12 * 60 / simulation_dt) # timestep to generate the snap profile

p_retail = np.mean(np.array(price_retail).reshape(
    -1, int(simulation_dt/(24*60/len(price_retail)))), axis = 1)
p_procure = np.mean(np.array(price_wholesale).reshape(
    -1, int(simulation_dt/(24*60/len(price_wholesale)))), axis = 1)
p_holding = np.mean(-np.array(price_pos_imb).reshape(
    -1, int(simulation_dt/(24*60/len(price_pos_imb)))), axis = 1)
p_stockout = np.mean(np.array(price_neg_imb).reshape(
    -1, int(simulation_dt/(24*60/len(price_neg_imb)))), axis = 1)

# Directory configuration

main_dir = os.path.dirname(__file__)

res_sum_dir = os.path.join(main_dir, results_folder, res_sum_folder)
try:
    os.makedirs(res_sum_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

case_name_dic = {}
res_case_dir_dic = {}
case_single_t_dir = {}
res_case_imp_dir = {}
for case in range(No_cases):
    if bid_type[case] == 0:
        bid_tp = "exp"
    elif bid_type[case] == 1:
        bid_tp = "CR"
    case_name_dic[case] = \
        res_case_folder % (No_players[case], simulation_length, bid_tp,
                           range_schd_load_frac[case][0] * 100,
                           range_schd_load_frac[case][1] * 100)
    res_case_dir_dic[case] = os.path.join(
        main_dir, results_folder, case_name_dic[case])
    case_single_t_dir[case] = os.path.join(
        res_case_dir_dic[case], res_case_snapshot_folder)
    try:
        os.makedirs(case_single_t_dir[case])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    res_case_imp_dir[case] = os.path.join(
        res_case_dir_dic[case], res_case_imp_folder)
    try:
        os.makedirs(res_case_imp_dir[case])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#==============================================================================
# Run Model
#==============================================================================
 
plt.ioff()

# dictionary[case][i]
indiv_l_exp = {}
schd_l_exp = {}
schd_l_frac = {}
schd_l_min = {}
schd_l_max = {}
schd_l_std_d = {}
schd_l_dist = {}
schd_l_trun_actl = {}
nonschd_l_exp = {}
nonschd_l_min = {}
nonschd_l_max = {}
nonschd_l_std_d = {}
nonschd_l_dist = {}
#nonschd_l_actl_list = {}

# dictionary[case]
schd_l_trun_actl_sum = {}
nonschd_l_exp_sum = {}
tot_l_exp_wo_data = {}
tot_l_exp_w_data = {}
tot_l_dist_wo_data = {}
tot_l_dist_w_data = {}

for case in range(No_cases):
    
    # All loads in [kWh]
    (indiv_l_exp[case], schd_l_exp[case], schd_l_frac[case], 
     schd_l_min[case], schd_l_max[case], schd_l_std_d[case], 
     schd_l_dist[case], schd_l_trun_actl[case], 
     nonschd_l_exp[case], nonschd_l_min[case], nonschd_l_max[case], 
     nonschd_l_std_d[case], nonschd_l_dist[case], schd_l_trun_actl_sum[case],
     nonschd_l_exp_sum[case], tot_l_exp_wo_data[case], tot_l_exp_w_data[case], 
     tot_l_dist_wo_data[case], tot_l_dist_w_data[case], t_list) = \
        data_initial(
            No_players[case], No_data_input, rand_p_sel, custom_p_sel,
            simulation_dt, simulation_length, simulation_start_time,
            load_profile_dt, res_case_dir_dic[case],
            case_name_dic[case], data_in_folder, load_dat_folder,
            cust_sel_folder, l_input_tp, uni_rand_floor, uni_rand_ceil, 
            generic_l_exp[case], range_schd_load_frac[case], 
            range_l_to_mean[case], schd_l_d_to_mean[case], 
            nonschd_l_d_to_mean[case], indiv_l_out, case_l_pdf_out, snap_tstep)
    
    if single_t_out == 1:
        prof_snap(
            snap_tstep, No_reals, No_quan_wholesale, No_players[case], 
            tot_l_dist_w_data[case], tot_l_dist_wo_data[case], 
            schd_l_min[case], nonschd_l_min[case],
            schd_l_max[case], nonschd_l_max[case],
            tot_l_exp_w_data[case], tot_l_exp_wo_data[case],
            p_retail[snap_tstep], p_procure[snap_tstep], 
            p_holding[snap_tstep], p_stockout[snap_tstep], t_list,
            simulation_dt, case_single_t_dir[case])
    
#==============================================================================
    
    (v_all_cl, sh_i, sh_ex_cl, sh_max_ex, nu_i, nu_ex_cl, nu_max_ex) = \
        coop_imp(
            No_players[case], simulation_length, simulation_dt, No_reals, 
            p_retail, p_procure, p_holding, p_stockout, bid_type[case], 
            tot_l_dist_w_data[case], tot_l_dist_wo_data[case],
            tot_l_exp_wo_data[case], schd_l_trun_actl[case], 
            schd_l_dist[case], nonschd_l_dist[case], 
            schd_l_exp[case], nonschd_l_exp[case], 
            schd_l_std_d[case], nonschd_l_std_d[case], indiv_l_exp[case],
            sc_v_def, res_case_imp_dir[case])
    
#==============================================================================


#==============================================================================
# Results Processing
#==============================================================================


#==============================================================================
# Results Output
#==============================================================================

