from cvxopt import matrix, solvers
from cvxopt.modeling import variable, op
solvers.options['show_progress'] = False
import sys

def imputation_min_excess(PLAYERS, IMP_PRE, KNOWN_EXCESS, V_DIC, ITER, \
                          B_SC_ITER, CT=0):
    # PLAYERS: list of player numbers, IMP_PRE: initialized imputation
    ## KNOWN_EXCESS: dictionary of known excesses keyed by subcoalition 
    ### e.g. {(1,): -0.31, (2,): -0.53, (1,2) : 0.}, 
    #### V_DIC: value dictionary keyed by subcoalition 
    ##### e.g. {(1,): 0., (2,): 0., (1,2) : 0.84}

#==============================================================================
#   Map Players to a list that starts from 1 and without interruption
#==============================================================================
    
    player_adj = {}
    i_adjust = 1
    for i in PLAYERS:
        player_adj[i] = i_adjust
        i_adjust += 1

#==============================================================================
#   Decision Variables
#==============================================================================
    
    num_vars = len(PLAYERS)
    allocation_vars = variable(num_vars,'allocation')
    excess_vars = variable(1,'largest excess')
    
#==============================================================================
#   Constraint 1: Equality Constraints
#==============================================================================
    
    # construct the matrices for the equality constraint
    # 'bimap_e' - binary matrix for subcoalitions with known excesses
    # 'v_vec_e' - vector of values of subcoalitions with known excesses
    # 'e_vec' - vector of excesses of subcoalitions with known excesses
    # 'sub_e' - list of subcoalitions with known excesses
    bimap_e = matrix(0., (len(KNOWN_EXCESS), num_vars))
    cnt_sub_e = 0
    sub_e = []
    v_e = []
    e_sub_e = []
    for sc in list(KNOWN_EXCESS):
        sub_e.append(sc)
        v_e.append(V_DIC[sc])
        e_sub_e.append(KNOWN_EXCESS[sc])
        for i in sc:
            bimap_e[cnt_sub_e, int(player_adj[i]-1)] = 1.
        cnt_sub_e += 1    
    v_vec_e = matrix(v_e)
    e_vec = matrix(e_sub_e)
    
    # equality constraint (relaxed): fix excess values for subcoalitions with
    ## known excesses calculated in previous iterations
    c1_1 = (v_vec_e - bimap_e * allocation_vars >= e_vec)
    c1_2 = (v_vec_e - bimap_e * allocation_vars <= e_vec + 1E-04)


#==============================================================================
#   Constraint 2: Inequality Constraint
#==============================================================================
    
    # construct the matrices for the inequality constraint
    # 'bimap_allo' - binary matrix for subcoalitions with unknown excesses
    # 'v_vec_en' - vector of values of subcoalitions with unknown excesses
    sub_en = [sc for sc in list(V_DIC) if sc not in sub_e]
    bimap_allo = matrix(0., (len(sub_en), num_vars))
    cnt_sub_en = 0
    v_en = []
    for sc in sub_en:
        v_en.append(V_DIC[sc])
        for i in sc:
            bimap_allo[cnt_sub_en, int(player_adj[i]-1)] = 1.
        cnt_sub_en += 1    
    v_vec_en = matrix(v_en)

    # inequality constraint: set upper bound of the unknown subcoalitional 
    ## excesses that are being optimized
    c2 = (v_vec_en - bimap_allo * allocation_vars <= excess_vars)
        
    
#==============================================================================
#   Objective Function
#==============================================================================
    
    prob = op(excess_vars, [c1_1, c1_2, c2])
    
    
#==============================================================================
#   Solve
#==============================================================================
    
    prob.solve()
    
    
#==============================================================================
#   Output
#==============================================================================
    
    known_exc_wo_gc = KNOWN_EXCESS.copy()
    known_exc_wo_gc.pop(tuple(PLAYERS), None)
    binding_sub = []
    if len(known_exc_wo_gc) != 0 \
    and prob.objective.value()[0] > min(known_exc_wo_gc.values()) + 1E-04:
        return (prob.objective.value()[0], IMP_PRE, binding_sub)
    else:
        try:
            # imputation as a dictionary with players as keys
            allocation = dict(zip(PLAYERS, allocation_vars.value))
            # find binding subcoalitions by identifying positive dual variables
            
            lagrange_mult_sum = 0
            binding_cstr_criterion = 1.0
            bd_cr = 0
            while lagrange_mult_sum == 0:
                lagrange_mult_sum = 0
                binding_cstr_criterion = binding_cstr_criterion*0.9
                for k in range(0, len(sub_en)):
                    if c2.multiplier.value[k] >= binding_cstr_criterion:
                        binding_sub.append(sub_en[k])
                        lagrange_mult_sum += c2.multiplier.value[k]
                bd_cr += 1
            #if bd_cr >= 2:
                #print("when binding_cstr_criterion = %f" \
                      #% binding_cstr_criterion)
                #print("binding coalitions are:")
                #print(binding_sub)
            # return prob.objective.value()[0]: excess value optimized,
            ## allocation: allocation dictionary keyed by player
            ### binding_sub: list of tuples of binding subcoalitions
            return (prob.objective.value()[0], allocation, binding_sub)
        except:
            
            ct_monitor = CT + 1
            if ct_monitor <= 0:
                return imputation_min_excess(
                        PLAYERS, KNOWN_EXCESS, V_DIC, ct_monitor) 
            else:
                print("these are the probablematic values")
                print(allocation_vars.value)
                print("known accesses:")
                print(list(KNOWN_EXCESS.values()))
                print(len(KNOWN_EXCESS))
                print("iter = %d" % ITER)
                print("Number of binding coalitions by iteration:")
                print(B_SC_ITER)
                sys.exit("nucleolus cannnot be computed for this case.")


###############################################################################
###############################################################################

def nucleolus(PLAYERS, VALUES):
    # PLAYERS: list of player numbers, 
    ## VALUES: value dictionary keyed by subcoalition 
    ### e.g. {(1,): 0., (2,): 0., (1,2) : 0.84}

#==============================================================================
#   Initialize Model Iteration Inputs
#==============================================================================
    
    # define grand coalition as a sorted tuple with its excess being zero
    grand_coalition = tuple(sorted(tuple(PLAYERS)))
    # initialize imputation by distributing gc value to all players evenly
    imputation_iter = dict(zip(PLAYERS, 
                               [VALUES[grand_coalition]] * len(PLAYERS)))
    # initialize the list of binding coalitions
    binding_subco_iter = [grand_coalition]
    gc_excess = 0.
    known_excess = {grand_coalition : gc_excess}
    excess = {grand_coalition : gc_excess}
    iter_count = 0
    max_iter = len(VALUES)
    b_sub_ct_iter = {}

#==============================================================================
#   Run Model Iteratively
#==============================================================================
    
    # iterate until the excesses for all subcoalitions have been determined
    while not set(list(VALUES)) <= set(list(excess)):
        # break out of while loop if the No. of iterations exceeds the No. of
        # coalitions, or no binding coalitions were found in the last iteration
        if iter_count >= max_iter or len(binding_subco_iter) == 0:
            for sc in list(VALUES):
                try:
                    excess[sc] = excess[sc]
                except:
                    exc = VALUES[sc]
                    for i in sc:
                        exc = exc - imputation_iter[i]
                    excess[sc] = exc
            break
        
        iter_count += 1
        
        (excess_iter, 
         imputation_iter, 
         binding_subco_iter) = imputation_min_excess(
                                     grand_coalition, imputation_iter, \
                                     known_excess, VALUES, \
                                     iter_count, b_sub_ct_iter)
        if len(binding_subco_iter) != 1:
            b_sub_ct_iter[iter_count] = len(binding_subco_iter)
            
        # update list of subcoalitions whose excesses were determined in the
        ## last iteration of optimisation
        for binding_subcoalition_1 in binding_subco_iter:
            known_excess.update({binding_subcoalition_1: excess_iter})
            
            # test if current binding subcoalition has already been recorded
            if binding_subcoalition_1 in list(excess):
                continue
            
            # adding unions of current binding sc with existing binding sc's
            ## and their conjugates
            for sc in list(excess):
                if any(x in list(binding_subcoalition_1) for x in list(sc)):
                    continue
                combined_binding_sc = tuple(
                        sorted(list(binding_subcoalition_1)+list(sc)))
                
                if combined_binding_sc not in list(VALUES):
                    continue
                if combined_binding_sc not in list(excess): 
                    excess.update(
                            {combined_binding_sc: excess_iter + excess[sc] \
                             - VALUES[binding_subcoalition_1] - VALUES[sc] \
                             + VALUES[combined_binding_sc]})
                combined_binding_sc_conj = tuple(
                        sorted([i for i in list(grand_coalition) \
                                if i not in list(combined_binding_sc)]))
                if combined_binding_sc_conj not in list(VALUES):
                    continue
                if combined_binding_sc_conj not in list(excess):
                    excess.update(
                            {combined_binding_sc_conj: \
                             VALUES[combined_binding_sc] \
                             + VALUES[combined_binding_sc_conj] \
                             - VALUES[grand_coalition] \
                             - excess[combined_binding_sc]})
    
            # adding current binding subcoalition and its conjugate
            excess.update({binding_subcoalition_1: excess_iter})
            binding_sc_1_conj = tuple(sorted([
                    i for i in list(grand_coalition) if 
                    i not in list(binding_subcoalition_1)]))
            if binding_sc_1_conj in list(VALUES) \
            and binding_sc_1_conj not in list(excess):
                excess.update({binding_sc_1_conj:
                    (VALUES[binding_subcoalition_1]+VALUES[binding_sc_1_conj]\
                     -VALUES[grand_coalition]-excess_iter)})
            
    # hide grand coalition excess first to find largest excess
    excess.pop(grand_coalition, None)
    e_max = max(excess.values())
    # put grand coalition excess back into the excess dictionary
    excess[grand_coalition] = 0.
    #print(excess)

    # return e_max: largest excess value optimized (in the first iteration),
    ## imputation_iter: final imputation dictionary keyed by player
    ### excess: dictionary of optimized excesses keyed by subcoalition, 
    #### iter_count: total number of iterations
    ##### b_sub_ct_iter：dictionary of number of binding coalitions per iter.
    return (e_max, imputation_iter, excess, iter_count, b_sub_ct_iter)