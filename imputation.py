import os
import numpy as np
import pandas as pd
#import csv
import matplotlib.pyplot as plt

from prof_calc import c_v_calc
from functions import dict_key_switch
from shapley_calc import shapley
from nucleolus_calc import nucleolus

#==============================================================================
#   Profit Calculation
#==============================================================================

    
def coop_imp(N, SIM_T, SIM_DT, REAL_N, P_RETAIL_LST, P_PROCURE_LST, 
             P_HOLD_LST, P_STOCKOUT_LST, BID_TP, 
             TOT_DIST_W_DATA, TOT_DIST_WO_DATA, TOT_EXP_WO_DATA, 
             SCHD_ACTL, SCHD_DIST, NONSCHD_DIST, SCHD_EXP, NONSCHD_EXP, 
             SCHD_STDD, NONSCHD_STDD, INDIV_EXP, C_V_DEF, CASE_IMP_DIR):
    
    # BASE_TP: type of bid used as baseline: '0' expectation; '1': cost ratio
    
###############################################################################
#   Data processing.
    
    # changeable inputs
    sh_t_out = 'shapley_timestep.csv'
    nu_t_out = 'nucleolus_timestep.csv'
    imp_sum_out = 'imputation_summary.csv'
    imp_t_fig_name = 'sh_nu_by_time.pdf'
    imp_fig_name = 'imputations.pdf'
    cost_fig_name = 'energy_cost.pdf'
    
    # Number of timesteps examined
    No_times = int(SIM_T * 24 * 60 / SIM_DT)
    schd_l_kw_avg = [sum(SCHD_ACTL[i])/(SIM_T*24) for i in range(1, N+1)]
    
###############################################################################
#   Compute the imputations
    
    sh_player = {}
    nu_player = {}
    energy_cost_base = {}
    t_pd = pd.DataFrame({'timestep\player': range(1, No_times+1)})
    sh_t_pd = pd.DataFrame(columns=[i for i in range(0, N+1)])
    nu_t_pd = pd.DataFrame(columns=[i for i in range(0, N+1)])
    for i in range(0, N+1):
        sh_player[i] = 0
        nu_player[i] = 0
        energy_cost_base[i] = 0
    for t in range(No_times):
        print('t = %d / %d' % (t, No_times))
        t_day = int(t % (24 * 60 / SIM_DT))
        opt_cost_ratio = (P_STOCKOUT_LST[t_day] - P_PROCURE_LST[t_day]) \
            / (P_STOCKOUT_LST[t_day] + P_HOLD_LST[t_day])
        
        # players including the retailer (indexed by 0)
        players = list(range(N+1))
        if BID_TP == 0:
            base_bids = TOT_EXP_WO_DATA[t]
        elif BID_TP == 1:
            base_bids = TOT_DIST_WO_DATA[t].ppf(opt_cost_ratio)
        (v_cl, base_prof) = \
            c_v_calc(
                players, base_bids, 
                dict_key_switch(SCHD_ACTL)[t],
                dict_key_switch(SCHD_DIST)[t], 
                dict_key_switch(SCHD_EXP)[t], 
                dict_key_switch(SCHD_STDD)[t], 
                dict_key_switch(NONSCHD_DIST)[t], 
                dict_key_switch(NONSCHD_EXP)[t], 
                dict_key_switch(NONSCHD_STDD)[t], 
                TOT_DIST_W_DATA[t], 
                P_RETAIL_LST[t_day], P_PROCURE_LST[t_day], 
                P_HOLD_LST[t_day], P_STOCKOUT_LST[t_day], 
                REAL_N, opt_cost_ratio, BID_TP, C_V_DEF)
        
        for i in range(1, N+1):
            energy_cost_base[i] += INDIV_EXP[i][t] * P_RETAIL_LST[t_day]
        energy_cost_base[0] -= base_prof
        
        (sh_all, sh_excess_dic, sh_excess_max) = \
            shapley(players, v_cl)
        (nu_excess_max, nu_all, nu_excess_dic, tot_iter, num_bind_sc_iter) = \
            nucleolus(players, v_cl)
        
        sh_t_pd = sh_t_pd.append(sh_all, ignore_index=True)
        nu_t_pd = nu_t_pd.append(nu_all, ignore_index=True)
        for i in range(0, N+1):
            sh_player[i] += sh_all[i]
            nu_player[i] += nu_all[i]
            
    #print(sh_player)
    
    imp_sum_pd = pd.DataFrame({'Player': list(range(0, N+1)),
                               'Energy Cost': list(energy_cost_base.values()),
                               'Shapley': list(sh_player.values()),
                               'Nucleolus': list(nu_player.values())})
    # axis=0 (1) means concatenate rows (columns)
    sh_t_pd_w_t = pd.concat([t_pd, sh_t_pd], axis=1)
    sh_t_pd_w_t.to_csv(
        os.path.join(CASE_IMP_DIR, sh_t_out), index = False, header=True)
    nu_t_pd_w_t = pd.concat([t_pd, nu_t_pd], axis=1)
    nu_t_pd_w_t.to_csv(
        os.path.join(CASE_IMP_DIR, nu_t_out), index = False, header=True)
    imp_sum_pd.to_csv(
        os.path.join(CASE_IMP_DIR, imp_sum_out), index = False, header=True)

###############################################################################
#   Plot the imputations by time

    imp_t_fig = plt.figure(figsize=(7,6))
    imp_t_fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.94, 
                                wspace=.1, hspace=.25)
    
#   Plot the Shapley by time
    sh_t_players = imp_t_fig.add_subplot(211)
    
    for i in range(1,N+1):
        sh_t_players.plot(t_pd['timestep\player'], sh_t_pd[i], lw=1, 
                          alpha=0.7, label='Player %d' % i)
    sh_t_players.yaxis.grid(b=True, color='gray', linestyle='dashed')
    sh_t_players.set_ylabel('Shapley value (£)', fontsize=13)
    sh_t_players.xaxis.set_ticks(np.arange(0, No_times+1, No_times/6))
    sh_t_players.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=10)
    box = sh_t_players.get_position()
    sh_t_players.set_position(
            [box.x0, box.y0 - box.height*0.105, box.width, box.height])
    
#   Plot the Nucleolus by time

    nu_t_players = imp_t_fig.add_subplot(212)
    
    for i in range(1,N+1):
        nu_t_players.plot(t_pd['timestep\player'], nu_t_pd[i], lw=1,
                          alpha=0.7, label='Player %d' % i)
    nu_t_players.yaxis.grid(b=True, color='gray', linestyle='dashed')
    nu_t_players.set_ylabel('Nucleolus (£)', fontsize=13)
    nu_t_players.set_xlabel('Timestep (%d min increments)' % SIM_DT, 
                            fontsize=13)
    nu_t_players.xaxis.set_ticks(np.arange(0, No_times+1, No_times/6))
    
    imp_t_fig.savefig(os.path.join(CASE_IMP_DIR, imp_t_fig_name), \
                    format = 'pdf', dpi = 1000)
    plt.show()
    
###############################################################################
#   Plot the imputations along with the realized schedulable loads

    player_label = ['Retailer']
    for i in range(1, N+1):
        player_label.append(i)
    
    # setting parameters for the following two plots
    ind = np.arange(N+1)
    width = 0.35
    
    imp_fig = plt.figure(figsize=(7,4))
    imp_fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, 
                                wspace=.1, hspace=.25)
    imp_players = imp_fig.add_subplot(111)
    
    sh_plt = imp_players.bar(ind-width/2, list(sh_player.values()), width, 
                             color='none', edgecolor='r', hatch='///',
                             label='Shapley')
    nu_plt = imp_players.bar(ind+width/2, list(nu_player.values()), width, 
                             color='none', edgecolor='b', hatch='///',
                             label='Nucleolus')
    imp_players.yaxis.grid(b=True, color='gray', linestyle='dashed')
    
    imp_players.set_ylabel('Payoff Allocation (£)', fontsize=13)
    imp_players.set_xlabel('Player', fontsize=13)
    imp_players.set_xticks(ind)
    imp_players.set_xticklabels(player_label, fontsize=10)
    #box = imp_players.get_position()
    #imp_players.set_position(
        #[box.x0, box.y0, box.width * 1.05, box.height*1.05])
    imp_players.legend(
        loc='upper right', bbox_to_anchor=(0.44, 1.12), ncol=2, fontsize=10)
    
    ec_players = imp_players.twinx()
    #ec_wo_retailer = list(energy_cost_base.values())
    #ec_wo_retailer.pop(0)
    #ec_players.plot(
        #ind, ec_wo_retailer, 'k-', lw=2, label='Baseline total energy cost')
    ec_players.plot(
        np.arange(1,N+1), schd_l_kw_avg, c='darkgreen',linestyle='--', 
        lw=1.5, marker='o', label='Average actual schedulable load')
    ec_players.set_ylim(bottom=0)
    ec_players.set_ylabel('Schedulable load (kW)', fontsize=13,
                          color='darkgreen')
    ec_players.tick_params(axis='y', labelcolor='darkgreen')
    ec_players.legend(
        loc='upper left', bbox_to_anchor=(0.49, 1.12), fontsize=10)
    imp_fig.savefig(os.path.join(CASE_IMP_DIR, imp_fig_name), \
                    format = 'pdf', dpi = 1000)
    plt.show()
    
###############################################################################
#   Plot the imputations in relation to the energy costs/profits
    
    cost_legend_label = ['EC w/o Coop.', 'EC w/ Shapley', 'Shapley value', 
                         'EC w/ Nucleolus', 'Nucleolus']
    ec_w_sh = np.subtract(list(energy_cost_base.values()), 
                          list(sh_player.values()))
    ec_w_nu = np.subtract(list(energy_cost_base.values()), 
                          list(nu_player.values()))
    cost_fig = plt.figure(figsize=(7,4))
    cost_fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.12, bottom=0.13, right=0.95, top=0.85, 
                                wspace=.1, hspace=.25)
    cost_players = cost_fig.add_subplot(111)
    
    ec_sh_plt = cost_players.bar(
        ind-width/2, ec_w_sh, width, color='coral', edgecolor='r', alpha=0.5)
    sh_plt = cost_players.bar(
        ind-width/2, list(sh_player.values()), width, bottom=ec_w_sh, 
        color='none', edgecolor='r', hatch='///')
    ec_nu_plt = cost_players.bar(
        ind+width/2, ec_w_nu, width, color='skyblue', edgecolor='b', alpha=0.5)
    nu_plt = cost_players.bar(
        ind+width/2, list(nu_player.values()), width, bottom=ec_w_nu, 
        color='none', edgecolor='b', hatch='///')
    
    cost_players.bar(
            ind, [0]*len(ind), width*2.5, 
            bottom=list(energy_cost_base.values()), linewidth=1.5,
            color='none', edgecolor='k')
    ec_leg_dummy = plt.Line2D((0,1),(0,0), linewidth=1.5, color='k')
    
    cost_players.yaxis.grid(b=True, color='gray', linestyle='dashed')
    
    cost_players.set_ylabel('Energy Cost (EC) (£)', fontsize=13)
    cost_players.set_xlabel('Player', fontsize=13)
    cost_players.set_xticks(ind)
    cost_players.set_xticklabels(player_label, fontsize=10)
    
    imp_legend_item = [ec_leg_dummy, ec_sh_plt[0], sh_plt[0],
                       ec_nu_plt[0], nu_plt[0]]
    cost_players.legend(
        imp_legend_item, cost_legend_label, loc='upper center', 
        bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=10)
    #imp_leg.get_frame().set_alpha(0.5)
    cost_fig.savefig(os.path.join(CASE_IMP_DIR, cost_fig_name), \
                    format = 'pdf', dpi = 1000)
    plt.show()

###############################################################################
    #   Results
    
    return (v_cl, sh_all, sh_excess_dic, sh_excess_max, 
            nu_all, nu_excess_dic, nu_excess_max)

