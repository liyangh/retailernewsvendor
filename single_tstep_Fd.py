import os
import numpy as np
#import csv
import matplotlib.pyplot as plt

from prof_calc import profit_calc

#==============================================================================
#   Profit Calculation
#==============================================================================

    
def prof_snap(SNAP_T, REAL_N, INCRE_N, N, TOT_DIST_W_DATA, TOT_DIST_WO_DATA,
              SCHD_MIN, UNSCHD_MIN, SCHD_MAX, UNSCHD_MAX, 
              TOT_EXP_W_DATA, TOT_EXP_WO_DATA, 
              P_RETAIL, P_PROCURE, P_HOLD, P_STOCKOUT, T_LST, S_DT, SNAP_DIR):
    
    # REAL_N: number of realizations of pdfs under each wholesale quantity,
    # INCRE_N: number of tested wholesale imports, evenly distributed over 
    # the possible wholesale quantities
    # SNAP_T: the time step that is used to produce the single-shot profit 
    # comparison figure
    
###############################################################################
#   Plot the snapshot pinball curve along with the load distributions.
    
    opt_cost_ratio = (P_STOCKOUT - P_PROCURE) / (P_STOCKOUT + P_HOLD)
    
    tot_l_actl_list = TOT_DIST_W_DATA[SNAP_T].rvs(REAL_N)
    quan_wholesale = np.linspace(
        sum([SCHD_MIN[i][SNAP_T] for i in range(1, N+1)]) + 
        sum([UNSCHD_MIN[i][SNAP_T] for i in range(1, N+1)]),
        sum([SCHD_MAX[i][SNAP_T] for i in range(1, N+1)]) + 
        sum([UNSCHD_MAX[i][SNAP_T] for i in range(1, N+1)]),
        INCRE_N)
    
    profit_by_bid = {}
    profit_by_bid_avg = {}
    for b in range(INCRE_N):
        (profit_by_bid_avg[b], profit_by_bid[b]) = \
            profit_calc(
                quan_wholesale[b], tot_l_actl_list, [], 
                P_RETAIL, P_PROCURE, P_HOLD, P_STOCKOUT, REAL_N)
            
    max_profit_ind = list(profit_by_bid_avg.values()).index(
                                max(profit_by_bid_avg.values()))
    max_profit_bid = quan_wholesale[max_profit_ind]
    
    
    best_bid_wo_data = TOT_DIST_WO_DATA[SNAP_T].ppf(opt_cost_ratio)
    best_bid_w_data = TOT_DIST_W_DATA[SNAP_T].ppf(opt_cost_ratio)
    print('At %s' % T_LST[SNAP_T])
    print('the wholesale purchase quantity that yields the highest profit: %f'
          % max_profit_bid)
    print('total load expectation without knowledge of controlable loads: %f'
          % TOT_EXP_WO_DATA[SNAP_T])
    print('purchase quantity on cost ratio w/o data on controlable loads: %f'
          % best_bid_wo_data)
    print('total load expectation with knowledge of controlable loads: %f'
          % TOT_EXP_W_DATA[SNAP_T])
    print('purchase quantity on cost ratio with data on controlable loads: %f'
          % best_bid_w_data)
    
    profit_dist_fig = plt.figure(figsize=(7,7))
    profit_dist_fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.13, bottom=0.13, right=0.89, top=0.92, \
                        wspace=.1, hspace=.25)
    profit_real = profit_dist_fig.add_subplot(211)
    for b in range(INCRE_N):
        profit_real.scatter(
            [quan_wholesale[b]] * REAL_N, profit_by_bid[b],
            c='g', alpha=0.1, s=2, marker='.')
        profit_real.scatter(
            quan_wholesale[b], profit_by_bid_avg[b],
            c='purple', marker='*', s=10)
    y_min, y_max = profit_real.get_ylim()
    profit_real.vlines(max_profit_bid, y_min, y_max, color='purple', 
                       linestyle='-.', lw=1.2, label='max avg. profit')
    leg_dummy = [
        profit_real.scatter(
            [], [], c='g', s=10, marker='.', label='profit per run'),
        profit_real.scatter(
            [], [], c='purple', marker='*', s=20, label='average profit')]
    profit_real.set_ylabel('Retailer Profit (£)', fontsize=13)
    profit_real.legend(loc='upper right', bbox_to_anchor=(0.57, 1.225), ncol=2)
    
    box = profit_real.get_position()
    profit_real.set_position(
            [box.x0, box.y0 - box.height*0.15, box.width, box.height*1.05])
        
    tot_l_dist_plt = profit_real.twinx()
    tot_l_dist_plt.plot(
        quan_wholesale, 
        TOT_DIST_WO_DATA[SNAP_T].pdf(quan_wholesale),
        'b--', lw=2, label='forecast load pdf w/o data')
    tot_l_dist_plt.plot(
        quan_wholesale, 
        TOT_DIST_W_DATA[SNAP_T].pdf(quan_wholesale), 
        'r--', lw=2, label='forecast load pdf w/ data')
    tot_l_dist_plt.set_ylabel('Probability Density', fontsize=13)
    tot_l_dist_plt.legend(loc='upper left', bbox_to_anchor=(0.57, 1.225), 
                          ncol=1)
    box = tot_l_dist_plt.get_position()
    tot_l_dist_plt.set_position(
            [box.x0, box.y0 - box.height*0.135, box.width, box.height*1.12])
    #profit_dist_fig.tight_layout()
    
    tot_l_cdf_plt = profit_dist_fig.add_subplot(212)
    tot_l_cdf_plt.plot(
        quan_wholesale, 
        TOT_DIST_WO_DATA[SNAP_T].cdf(quan_wholesale),
        'b-', lw=2, label='forecast load cdf w/o data')
    tot_l_cdf_plt.plot(
        quan_wholesale, 
        TOT_DIST_W_DATA[SNAP_T].cdf(quan_wholesale), 
        'r-', lw=2, label='forecast load cdf w/ data')
    y_min, y_max = tot_l_cdf_plt.get_ylim()
    tot_l_cdf_plt.hlines(opt_cost_ratio, 
                         0, max(TOT_DIST_WO_DATA[SNAP_T].ppf(opt_cost_ratio), 
                                TOT_DIST_W_DATA[SNAP_T].ppf(opt_cost_ratio)),
                         color='darkgoldenrod', linestyle='-.', lw=1.2,
                         label='optimal cost ratio')
    tot_l_cdf_plt.scatter(TOT_DIST_WO_DATA[SNAP_T].ppf(opt_cost_ratio), 
                       opt_cost_ratio, color='b', marker='o', s=30)
    tot_l_cdf_plt.scatter(TOT_DIST_W_DATA[SNAP_T].ppf(opt_cost_ratio), 
                       opt_cost_ratio, color='r', marker='o', s=30)
    tot_l_cdf_plt.vlines(max_profit_bid, y_min, y_max, color='purple', 
                         linestyle='-.', lw=1.2)
    tot_l_cdf_plt.set_xlabel(
        'Procured Wholesale Energy (kWh) - %d min Duration' % S_DT, 
        fontsize=13)
    tot_l_cdf_plt.set_ylabel('Cumulative Probability', fontsize=13)
    tot_l_cdf_plt.legend(loc='best', ncol=1)
    box = tot_l_cdf_plt.get_position()
    tot_l_cdf_plt.set_position(
            [box.x0, box.y0 - box.height*0.16, box.width, box.height*1.12])
    
    plt.show()
    profit_dist_fig_name = "profit_distribution.pdf"
    profit_dist_fig.savefig(
        os.path.join(SNAP_DIR, profit_dist_fig_name),
        format = 'pdf', dpi = 1000)
    
###############################################################################
#   Plot the snapshot violin curves of the profit distribution based on 
#   special wholesale quantities
    
    bids_list = [TOT_EXP_WO_DATA[SNAP_T], best_bid_wo_data, 
                 TOT_EXP_WO_DATA[SNAP_T], best_bid_wo_data, 
                 TOT_EXP_W_DATA[SNAP_T], best_bid_w_data]
    l_knw_r_list = [0] * 2 + [tot_l_actl_list] * 4
    l_unk_dis_list = [[TOT_DIST_WO_DATA[SNAP_T]]] * 2 + [[]] * 4
    labels = ['Exp. w/o data', 
              '\n\nLoad realized w/ forecast \nWITHOUT schd load data',
              'CR w/o data', 
              'Exp. w/o data', 
              'CR w/o data', 
              '\n\nLoad realized w/ forecast \nWITH schedulable load data',
              'Exp. w/ data', 
              'CR w/ data']
    
    profit_by_b_sp = {}
    profit_by_b_sp_avg = {}
    for b_sp in range(len(bids_list)):
        (profit_by_b_sp_avg[b_sp], profit_by_b_sp[b_sp]) = \
            profit_calc(
                bids_list[b_sp], l_knw_r_list[b_sp], l_unk_dis_list[b_sp], 
                P_RETAIL, P_PROCURE, P_HOLD, P_STOCKOUT, REAL_N)
    
    ind = np.arange(1, len(bids_list) + 1)
    data = [sorted(p_list) for p_list in list(profit_by_b_sp.values())]
    p_means = list(profit_by_b_sp_avg.values())
    
    profit_b_sp_fig = plt.figure(figsize=(7,4))
    profit_b_sp_fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.12, bottom=0.26, right=0.97, top=0.91, \
                        wspace=.1, hspace=.25)
    profit_real_sp = profit_b_sp_fig.add_subplot(111)
    parts = profit_real_sp.violinplot(
            data, showextrema=False, showmeans=False, showmedians=False)
    
    for pc in range(len(parts['bodies'])):
        if pc <= 1:
            parts['bodies'][pc].set_facecolor('mediumaquamarine')
        else:
            parts['bodies'][pc].set_facecolor('violet')
        #pc.set_edgecolor('black')
        parts['bodies'][pc].set_alpha(0.5)
    
    quartile1, quartile3 = np.percentile(data, [25, 75], axis=1)
    quantile_05, quantile_95 = np.percentile(data, [5, 95], axis=1)
    
    profit_real_sp.scatter(
        ind, p_means, marker='o', color='white', edgecolor='k', 
        s=30, zorder=3, label='Mean')
    profit_real_sp.vlines(
        ind, quartile1, quartile3, color='k', linestyle='-', lw=5, 
        label='Quartiles')
    profit_real_sp.vlines(
        ind, quantile_05, quantile_95, color='k', linestyle='-', lw=1, 
        label='5%, 95% quantiles')
    
    # set style for the axes
    
    profit_real_sp.set_xticks([1, 1.5, 2, 3, 4, 4.5, 5, 6], minor = False)
    profit_real_sp.set_xticklabels(labels)
    profit_real_sp.set_xticks([0.5, 2.5, 6.5], minor = True)
    profit_real_sp.set_xlabel(
        'Method to Choose Quantity of Procured Wholesale Energy', fontsize=13)
    profit_real_sp.yaxis.grid(b=True, color='gray', linestyle='dashed')
    profit_real_sp.set_ylabel('Retailer Profit (£)', fontsize=13)
    profit_real_sp.tick_params(
        axis='x', which='minor', direction='out', length=50)
    profit_real_sp.tick_params(
        axis='x', which='major', bottom=False, top=False)
    profit_real_sp.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3)
    
    plt.show()
    profit_b_sp_fig_name = "profit_distribution_specific.pdf"
    profit_b_sp_fig.savefig(
        os.path.join(SNAP_DIR, profit_b_sp_fig_name),
        format = 'pdf', dpi = 1000)



###############################################################################
    #   Results
    
    return

