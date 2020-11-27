import os
import numpy as np
#import csv
import itertools
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt

from functions import all_combo

#==============================================================================
#   Profit Calculation
#==============================================================================

    
def profit_calc(PROC, KNOWN_DEM_R, UNKNOWN_DEM_DIST_LST, P_RETAIL, P_PROCURE, 
                P_HOLD, P_STOCKOUT, NO_R):
    # PROC, KNOWN_DEM_R, UNKNOWN_DEM_DIST_LST: kWh
    # P_X: £/kWh
    # NO_R: Number of runs
    
###############################################################################
#   Calculate the profit based on the bid and forecast distribution
    # Both known_demand_R and unknown_demand_R are lists keyed by R
    
    if type(KNOWN_DEM_R) is (int or float):
        known_demand_R = [KNOWN_DEM_R] * NO_R
    else:
        known_demand_R = KNOWN_DEM_R
        
    if UNKNOWN_DEM_DIST_LST == []:
        unknown_demand_R = [0] * NO_R
    else:
        unknown_demand_real = [dist.rvs(NO_R) for dist in UNKNOWN_DEM_DIST_LST]
        unknown_demand_R = [sum(dem) for dem in zip(*unknown_demand_real)]
        
    demand_actl_R = \
        [sum(dem) for dem in zip(*[known_demand_R, unknown_demand_R])]
    
    profit_lst = \
        [dem_actl * P_RETAIL 
         - PROC * P_PROCURE
         - max(PROC - dem_actl, 0) * P_HOLD
         - max(dem_actl - PROC, 0) * P_STOCKOUT
         for dem_actl in demand_actl_R]
        
    profit_avg = sum(profit_lst)/NO_R

###############################################################################
    #   Results
    
    # profit in £
    
    return (profit_avg, profit_lst)

#==============================================================================
#   Coalition Value Calculation
#==============================================================================

def c_v_calc(PLAYERS, B_BASE, SCHD_L_REAL, SCHD_L_DIST, SCHD_L_EXP, SCHD_L_STD, 
             NONSCHD_L_DIST, NONSCHD_L_EXP, NONSCHD_L_STD, TOT_L_DIST_W_DATA, 
             P_RETAIL, P_PROCURE, P_HOLD, P_STOCKOUT, NO_R, CR, B_TYPE, 
             CLTN_V_DEF):
    #TOT_L_DIST_W_DATA: the distribution of the load sum of all the players
    
    N = PLAYERS[-1]
    coalitions = all_combo(PLAYERS)
    c_value = {}
    
    (prof_base_avg,
     prof_base) = profit_calc(B_BASE, 0, [TOT_L_DIST_W_DATA], 
                                 P_RETAIL, P_PROCURE,
                                 P_HOLD, P_STOCKOUT, NO_R)
    
    for cltn in coalitions:
        if 0 not in cltn:
            c_value[cltn] = 0 
        else:
            cltn_wo_0 = [i for i in cltn if i != 0]
            non_cltn = \
                [i for i in range(1, N+1) if i not in cltn]
            kwn_schd_l = sum(SCHD_L_REAL[i] for i in cltn_wo_0)
            
            unk_schd_l_exp = sum(SCHD_L_EXP[i] for i in non_cltn)
            unk_schd_var = sum(SCHD_L_STD[i]**2 for i in non_cltn)
            tot_l_exp_cltn = sum(NONSCHD_L_EXP.values()) + kwn_schd_l \
                                                        + unk_schd_l_exp
            tot_l_w_cltn_dis = norm(
                loc = tot_l_exp_cltn,
                scale = (sum(dev**2 for dev in NONSCHD_L_STD.values()) \
                         + unk_schd_var) ** (1/2))
            if B_TYPE == 0:
                bid_cltn = tot_l_exp_cltn
            elif B_TYPE == 1:
                bid_cltn = tot_l_w_cltn_dis.ppf(CR)
            
            if CLTN_V_DEF == 1:
                (prof_CF_cltn_avg,
                 prof_CF_bid_cltn) = profit_calc(bid_cltn, 0, 
                                                 [tot_l_w_cltn_dis], 
                                                 P_RETAIL, P_PROCURE,
                                                 P_HOLD, P_STOCKOUT, NO_R)
                (prof_base_cltn_avg,
                 prof_base_cltn) = profit_calc(B_BASE, 0, [tot_l_w_cltn_dis], 
                                               P_RETAIL, P_PROCURE,
                                               P_HOLD, P_STOCKOUT, NO_R)
                c_value[cltn] = prof_CF_cltn_avg - prof_base_cltn_avg
                
            elif CLTN_V_DEF == 0:
                (prof_CF_cltn_avg,
                 prof_CF_bid_cltn) = profit_calc(bid_cltn, 0, 
                                                 [TOT_L_DIST_W_DATA], 
                                                 P_RETAIL, P_PROCURE,
                                                 P_HOLD, P_STOCKOUT, NO_R)
                c_value[cltn] = prof_CF_cltn_avg - prof_base_avg
            
                
###############################################################################
    #   Results
    
    # profit in £
    
    return (c_value, prof_base_avg)