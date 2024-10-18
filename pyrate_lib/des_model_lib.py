#from numpy import *
import numpy as np
import scipy
import csv, os
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import random as rand
from itertools import *
from scipy.special import gdtr, gdtrix

def powerset(iterable):
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def build_rho_index_vec(obs_state,nareas,possible_areas,verbose=0):
    #verbose=1
    obs_state=set(obs_state)
    r_vec_index=[]
    r_neg_index=np.zeros((len(possible_areas),nareas))
    if verbose ==1: print("anc_state\tobs_state\ttemp\tr_neg")
    for i in range(len(possible_areas)):
        anc_state= set(possible_areas[i])
        if 1>2: pass #len(anc_state)==0: temp = np.repeat(0,nareas)
        else:
            temp=[]
            for j in range(nareas): 
                if j in anc_state and j in obs_state: 
                #   temp.append(nareas+1) # nareas+1 is index of r_vec = 1 
                    temp.append(j+1) # j+1 should be the index of r_{j}
                    r_neg_index[i,j]=1 # this will transform it in (1 - r_{j})

                elif j in anc_state and j not in obs_state: # area present in anc_states but not observed
                #   temp.append(nareas+1)
                    temp.append(j+1)  # j+1 should be the index of r_{j}
                elif j not in anc_state and j not in obs_state: # area absent in anc_states AND not observed
                    #temp.append(j+1) # j+1 should be the index of r_{j}
                    #r_neg_index[i,j]=1 # this will transform it in (1 - r_{j})
                    temp.append(nareas+1) # this option replaces (1 - r_{j}) with 1 | nareas+1 is index of r_vec = 1 
                elif j not in anc_state and j in obs_state:
                    temp.append(0) # 0 is index of r_vec = 0
                else: print("Warning: problem in function <build_rho_index_vec>")
        if verbose ==1: 
            r_vec= np.array([0]+list(np.zeros(nareas)+0.25) +[1])
            r_vec[1]=0.33
            print(anc_state,"\t",obs_state,"\t",temp,"\t",r_neg_index[i], abs(r_neg_index[i]-r_vec[temp]),np.prod(abs(r_neg_index[i]-r_vec[temp])))
        r_vec_index.append(temp)
    #print "\nFINAL",np.array(r_vec_index),r_neg_index    , "\nEND"
    #quit()
    return np.array(r_vec_index),r_neg_index

def build_list_rho_index_vec(obs_area_series,nareas,possible_areas,verbose=0):
    r_vec_indexes= list()
    sign_list= list()
    for n in obs_area_series:
        r_ind,sign= build_rho_index_vec(n,nareas,possible_areas,verbose)  # this is fixed (but different) for each node
        r_vec_indexes.append(r_ind)
        sign_list.append(sign)
        #print n,r_ind,sign
    return r_vec_indexes,sign_list

def make_Q(dv,ev): # construct Q matrix
    D=0
    [d1,d2] = dv # d1 A->B; d2 B->A;
    [e1,e2] = ev
    Q= np.array([
        [D, 0, 0, 0 ],
        [e1,D, 0, d1],
        [e2,0, D, d2],
        [0 ,e2,e1,D ]
    ])
    # fill diagonal values
    np.fill_diagonal(Q, -np.sum(Q,axis=1))
    return Q

def make_Q_list(dv_list,ev_list): # construct list of Q matrices
    Q_list=[]
    #print len(dv_list), dv_list
    for i in range(len(dv_list)):
        D=0
        [d1,d2] = dv_list[i] # d1 A->B; d2 B->A;
        [e1,e2] = ev_list[i]
        Q= np.array([
            [D, 0, 0, 0 ],
            #[D, d1, d2, 0 ],
            [e1,D, 0, d1],
            [e2,0, D, d2],
            [0 ,e2,e1,D ]
        ])
        # fill diagonal values
        np.fill_diagonal(Q, -np.sum(Q,axis=1))
        Q_list.append(Q)
    return Q_list

def make_Q_Covar(dv_list,ev_list,time_var,covar_par=np.zeros(2)): # construct list of Q matrices
    transf_d = np.array([dv_list[0][0] * np.exp(covar_par[0]*time_var), dv_list[0][1] * np.exp(covar_par[0]*time_var)]).T
    transf_e = np.array([ev_list[0][0] * np.exp(covar_par[1]*time_var), ev_list[0][1] * np.exp(covar_par[1]*time_var)]).T
    Q_list=[]
    for i in range(len(transf_d)):
        D=0
        [d1,d2] = transf_d[i] # d1 A->B; d2 B->A;
        [e1,e2] = transf_e[i]
        Q= np.array([
            [D, 0, 0, 0 ],
            #[D, d1, d2, 0 ],
            [e1,D, 0, d1],
            [e2,0, D, d2],
            [0 ,e2,e1,D ]
        ])
        # fill diagonal values
        np.fill_diagonal(Q, -np.sum(Q,axis=1))
        Q_list.append(Q)
    return Q_list

def make_Q_Covar4V(dv_list,ev_list,time_var,covar_par=np.zeros(4)): # construct list of Q matrices
    transf_d = np.array([dv_list[0][0] * np.exp(covar_par[0]*time_var), dv_list[0][1] * np.exp(covar_par[1]*time_var)]).T
    transf_e = np.array([ev_list[0][0] * np.exp(covar_par[2]*time_var), ev_list[0][1] * np.exp(covar_par[3]*time_var)]).T
    Q_list=[]
    for i in range(len(transf_d)):
        D=0
        [d1,d2] = transf_d[i] # d1 A->B; d2 B->A;
        [e1,e2] = transf_e[i]
        Q= np.array([
            [D, 0, 0, 0 ],
            # [D, d1, d2, 0 ],
            [e1,D, 0, d1],
            [e2,0, D, d2],
            [0 ,e2,e1,D ]
        ])
        # fill diagonal values
        np.fill_diagonal(Q, -np.sum(Q,axis=1))
        Q_list.append(Q)
    return Q_list

def transform_rate_logistic(r_at_trait_mean,prm,trait):
    # r0 is the max rate
    k, x0 = prm # steepness and mid point
    trait_mean = np.mean(trait)
    r0 = r_at_trait_mean * ( 1. +  np.exp( -k * (trait_mean-x0) )    )
    rate_at_trait = r0 / ( 1. +  np.exp( -k * (trait-x0) )    )
    return rate_at_trait.flatten()


def get_dispersal_rate_through_time(dv_list,time_var_d1,time_var_d2,covar_par=np.zeros(4),x0_logistic=np.zeros(4),transf_d=0): 
    if transf_d==1: # exponential
        transf_d = np.array([dv_list[0][0] * np.exp(covar_par[0]*time_var_d1), dv_list[0][1] * np.exp(covar_par[1]*time_var_d2)]).T
    elif transf_d==2: # logistic
        transf_d12 = transform_rate_logistic(dv_list[0][0], [covar_par[0],x0_logistic[0]],time_var_d1)
        transf_d21 = transform_rate_logistic(dv_list[0][1], [covar_par[1],x0_logistic[1]],time_var_d2)
        transf_d = np.array([transf_d12,transf_d21]).T
    else: # time-dependent-dispersal
        transf_d = dv_list
    return transf_d



def make_Q_Covar4VDdE(dv_list, ev_list, rep_d, rep_e,
                      time_var_d1, time_var_d2, time_var_e1, time_var_e2,
                      diversity_d1, diversity_d2, diversity_e1, diversity_e2, dis_into_1, dis_into_2,
                      covar_par=np.zeros(4), covar_parD=np.zeros(4), covar_parE=np.zeros(4),
                      x0_logisticD=np.zeros(4), x0_logisticE=np.zeros(4),
                      transf_d=0, transf_e=0,
                      offset_dis_div1=0, offset_dis_div2=0, offset_ext_div1=0, offset_ext_div2=0):
    if transf_d == 1: # exponential
        idx1 = np.arange(0, len(covar_parD), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parD), 2, dtype=int)
        d12 = dv_list[0][0] * np.exp(np.sum(covar_parD[idx1] * time_var_d1, axis=1))
        d21 = dv_list[0][1] * np.exp(np.sum(covar_parD[idx2] * time_var_d2, axis=1))
    elif transf_d == 2: # logistic
        d12 = transform_rate_logistic(dv_list[0][0], [covar_parD[0], x0_logisticD[0]], time_var_d1)
        d21 = transform_rate_logistic(dv_list[0][1], [covar_parD[1], x0_logisticD[1]], time_var_d2)
    elif transf_d == 4: # linear diversity dependence
        d12 = (dv_list[0][0] / (1. - (offset_dis_div1 / covar_par[0]))) * (1. - (diversity_d1 / covar_par[0]))
        d21 = (dv_list[0][1] / (1. - (offset_dis_div2 / covar_par[1]))) * (1. - (diversity_d2 / covar_par[1]))
        d12[transf_d12 <= 0] = 1e-5
        d12[np.isnan(transf_d12)] = 1e-5
        d21[transf_d21 <= 0] = 1e-5
        d21[np.isnan(transf_d21)] = 1e-5
    elif transf_d == 5: # Combination of environment and diversity dependent dispersal
        idx1 = np.arange(0, len(covar_parD), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parD), 2, dtype=int)
        env_d12 = dv_list[0][0] * np.exp(np.sum(covar_parD[idx1] * time_var_d1, axis=1))
        env_d21 = dv_list[0][1] * np.exp(np.sum(covar_parD[idx2] * time_var_d2, axis=1))
        d12 = (env_d12 / (1. - (offset_dis_div1/covar_par[0]))) * (1. - (diversity_d1/covar_par[0]))
        d21 = (env_d21 / (1. - (offset_dis_div2/covar_par[1]))) * (1. - (diversity_d2/covar_par[1]))
        d12[d12 <= 0] = 1e-5
        d12[np.isnan(d12)] = 1e-5
        d21[transf_d21 <= 0] = 1e-5
        d21[np.isnan(d21)] = 1e-5
    elif transf_d == 8: # exponential environmental dependence with rate shifts
        idx1 = np.arange(0, len(covar_parD), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parD), 2, dtype=int)
        d12 = dv_list[:, 0] * np.exp(np.sum(covar_parD[idx1] * time_var_d1, axis=None))
        d21 = dv_list[:, 1] * np.exp(np.sum(covar_parD[idx2] * time_var_d2, axis=None))
    else: # time-dependent-dispersal
        d12 = dv_list[:, 0]
        d21 = dv_list[:, 1]
    if transf_e == 1: # exponential
        idx1 = np.arange(0, len(covar_parE), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parE), 2, dtype=int)
        e1 = ev_list[0][0] * np.exp(np.sum(covar_parE[idx1] * time_var_e1, axis=1))
        e2 = ev_list[0][1] * np.exp(np.sum(covar_parE[idx2] * time_var_e2, axis=1))
    elif transf_e == 2: # logistic
        e1 = transform_rate_logistic(ev_list[0][0], [covar_parE[0], x0_logisticE[0]], time_var_e1)
        e2 = transform_rate_logistic(ev_list[0][1], [covar_parE[1], x0_logisticE[1]], time_var_e2)
    elif transf_e == 3: # linear dependence on dispersal fraction
        e1 = ev_list[0][0] + covar_par[2] * dis_into_1
        e2 = ev_list[0][1] + covar_par[3] * dis_into_2
        e1[e1 < 0.0]  = 1e-5
        e2[e2 < 0.0]  = 1e-5
    elif transf_e == 4: # linear diversity dependence
        base_e1 = ev_list[0][0] * (1. - (offset_ext_div1 / covar_par[2]))
        base_e2 = ev_list[0][1] * (1. - (offset_ext_div2 / covar_par[3]))
        denom_e1 = 1. - diversity_e1 / covar_par[2]
        denom_e2 = 1. - diversity_e2 / covar_par[3]
        denom_e1[denom_e1 == 0.0] = 1e-5 # Diversity equals K
        denom_e2[denom_e2 == 0.0] = 1e-5
        e1 = base_e1 / denom_e1
        e2 = base_e2 / denom_e2
        # Replace negative and infinite extinction rate when observed diversity is >= K by max extinction
        rep_e1 = base_e1 / (1. - ((covar_par[2] - 1e-5) / covar_par[2]))
        rep_e2 = base_e2 / (1. - ((covar_par[3] - 1e-5) / covar_par[3]))
        e1[e1 < 0.0] = rep_e1
        e1[np.isfinite(e1)] = rep_e1
        e2[e2 < 0.0] = rep_e2
        e2[np.isfinite(e2)] = rep_e2
    elif transf_e == 5: # Combination of environment and diversity dependent extinction
        idx1 = np.arange(0, len(covar_parE), 2, dtype = int)
        idx2 = np.arange(1, len(covar_parE), 2, dtype = int)
        env_e1 = ev_list[0][0] * np.exp(np.sum(covar_parE[idx1]*time_var_e1, axis = 1))
        env_e2 = ev_list[0][1] * np.exp(np.sum(covar_parE[idx2]*time_var_e2, axis = 1))
        denom_e1 = 1. - diversity_e1/covar_par[2]
        denom_e2 = 1. - diversity_e2/covar_par[3]
        denom_e1[denom_e1 == 0.0] = 1e-5 # Diversity equals K
        denom_e2[denom_e2 == 0.0] = 1e-5
        transf_e = np.array([env_e1 / denom_e1, env_e2 / denom_e2]).T
        # Replace negative and infinite extinction rate when observed diversity is >= K by max extinction
        rep_e1 = env_e1 / (1. - ((covar_par[2] - 1e-5) / covar_par[2]))
        rep_e2 = env_e2 / (1. - ((covar_par[3] - 1e-5) / covar_par[3]))
        idx_smaller0 = e1 < 0
        e1[idx_smaller0] = rep_e1[idx_smaller0]
        idx_na = np.isfinite(e1) == False
        e1[idx_na] = rep_e1[idx_na]
        idx_smaller0 = e2 < 0
        e2[idx_smaller0] = rep_e2[idx_smaller0]
        idx_na = np.isfinite(e2) == False
        e2[idx_na] = rep_e2[idx_na]
    elif transf_e == 6: # linear dependence on environment
        idx1 = np.arange(0, len(covar_parE), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parE), 2, dtype=int)
        e1 = ev_list[0][0] + np.sum(covar_parE[idx1] * time_var_e1, axis=1)
        e2 = ev_list[0][1] + np.sum(covar_parE[idx2] * time_var_e2, axis=1)
        e1[e1 < 0.0]  = 1e-5
        e2[e2 < 0.0]  = 1e-5
    elif transf_e == 7: # Combination of environment and dispersal dependent extinction
        idx1 = np.arange(0, len(covar_parE), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parE), 2, dtype=int)
        env_e1 = ev_list[0][0] * np.exp(np.sum(covar_parE[idx1]*time_var_e1, axis = 1))
        env_e2 = ev_list[0][1] * np.exp(np.sum(covar_parE[idx2]*time_var_e2, axis = 1))
        e1 = env_e1 + covar_par[2] * dis_into_1
        e2 = env_e2 + covar_par[3] * dis_into_2
        e1[e1 < 0.0]  = 1e-5
        e2[e2 < 0.0]  = 1e-5
    elif transf_e == 8: # exponential environmental dependence with rate shifts
        idx1 = np.arange(0, len(covar_parE), 2, dtype=int)
        idx2 = np.arange(1, len(covar_parE), 2, dtype=int)
        e1 = ev_list[:, 0] * np.exp(np.sum(covar_parE[idx1] * time_var_e1, axis=None))
        e2 = ev_list[:, 1] * np.exp(np.sum(covar_parE[idx2] * time_var_e2, axis=None))
    else:
        e1 = ev_list[:, 0]
        e2 = ev_list[:, 1]

    QT_array = np.zeros((max(len(d12), len(e1)), 4, 4))
    QT_array[:, 3, 1] = d12
    QT_array[:, 3, 2] = d21
    QT_array[:, 0, 1] = e1
    QT_array[:, 2, 3] = e1
    QT_array[:, 0, 2] = e2
    QT_array[:, 1, 3] = e2
#    col_sum = -np.einsum('ijk->ik', QT_array) # Colsum per slice
#    s0,s1,s2 = QT_array.shape
#    QT_array.reshape(s0,-1)[:,::s2+1] = col_sum
    
    d_rates = np.array([d12, d21]).T
    e_rates = np.array([e1, e2]).T
    if transf_d == 0:
        d_rates = d_rates[rep_e, :]
    if transf_e == 0:
        e_rates = e_rates[rep_d, :]
    return QT_array, [d_rates, e_rates]


def make_Q_Covar4VDdEDOUBLE(dv_list, ev_list, time_var_d1, time_var_d2, time_var_e1, time_var_e2, time_var_e1two, time_var_e2two,
                            covar_par=np.zeros(4), x0_logistic=np.zeros(4), transf_d=0, transf_e=0):
    if transf_d==1: # exponential
        transf_d = np.array([dv_list[0][0] * np.exp(covar_par[0]*time_var_d1), dv_list[0][1] * np.exp(covar_par[1]*time_var_d2)]).T
    elif transf_d==2: # logistic
        transf_d12 = transform_rate_logistic(dv_list[0][0], [covar_par[0],x0_logistic[0]],time_var_d1)
        transf_d21 = transform_rate_logistic(dv_list[0][1], [covar_par[1],x0_logistic[1]],time_var_d2)
        transf_d = np.array([transf_d12,transf_d21]).T
    else: # time-dependent-dispersal
        transf_d = dv_list
    if transf_e==1: # exponential
        transf_e = np.array([ev_list[0][0] * np.exp(covar_par[2]*time_var_e1 + covar_par[3]*time_var_e1two), ev_list[0][1] * np.exp(covar_par[2]*time_var_e2 + covar_par[3]*time_var_e2two)]).T
    elif transf_e==3: # linear DOUBLE
        transf_e = np.array([ev_list[0][0] + (covar_par[2]*time_var_e1) + (covar_par[3]*time_var_e1two) , ev_list[0][1] +(covar_par[2]*time_var_e2) + (covar_par[3]*time_var_e2two) ]).T
        transf_e[transf_e<0.0001] = 0.0001
    elif transf_e==2: # logistic
        transf_e1  = transform_rate_logistic(ev_list[0][0], [covar_par[2],x0_logistic[2]],time_var_e1)
        transf_e2  = transform_rate_logistic(ev_list[0][1], [covar_par[3],x0_logistic[3]],time_var_e2)
        transf_e = np.array([transf_e1 ,transf_e2 ]).T
    else:
        transf_e = ev_list

    Q_list=[]
    for i in range(len(transf_d)):
        D=0
        [d1,d2] = transf_d[i] # d1 A->B; d2 B->A;
        [e1,e2] = transf_e[i]
        Q= np.array([
            [D, 0, 0, 0 ],
            # [D, d1, d2, 0 ],
            [e1,D, 0, d1],
            [e2,0, D, d2],
            [0 ,e2,e1,D ]
        ])
        # fill diagonal values
        np.fill_diagonal(Q, -np.sum(Q,axis=1))
        Q_list.append(Q)
    return Q_list, [transf_d,transf_e]


def make_Q3A(dv,ev): # construct Q matrix
    D=0
    [d1,d2,d3] = dv # d1 A<->B; d2 B<->C; d3 A<->C
    [e1,e2,e3] = ev
    Q= np.array([
        # 0   A   B   C   AB  BC  AC  ABC
        [ D,  0,  0,  0,  0,  0,  0,  0    ],  # 0
        [e1,  D,  0,  0, d1,  0, d3,  0    ],  # A
        [e2,  0,  D,  0, d1,  d2, 0,  0    ],  # B
        [e3,  0,  0,  D,  0,  d2,d3,  0    ],  # C
        [ 0, e2, e1,  0,  D,  0,  0,  d3+d2],  # AB
        [ 0,  0, e3, e2,  0,  D,  0,  d1+d3],  # BC
        [ 0, e3,  0, e1,  0,  0,  D,  d1+d2],  # AC
        [ 0,  0,  0,  0, e3, e1, e2,  D    ],  # ABC
    ])
    # fill diagonal values
    np.fill_diagonal(Q, -np.sum(Q,axis=1))
    return Q



######################################
########      SIMULATOR      #########
######################################
def simulate_dataset(no_sim,d,e,n_taxa,TimeSpan,n_bins=20,wd=""):
    n_bins +=1
    def random_choice_P(vector):
        probDeath=np.cumsum(vector/np.sum(vector)) # cumulative prob (used to randomly sample one 
        r=rand.random()                          # parameter based on its deathRate)
        probDeath=np.sort(append(probDeath, r))
        ind=np.where(probDeath==r)[0][0] # just in case r==1
        return [vector[ind], ind]
    
    Q = make_Q(d,e)
    D = -np.diagonal(Q) # waiting times
    
    outfile="%s/sim_%s_%s_%s_%s_%s_%s.txt" % (wd,no_sim,n_taxa,d[0],d[1],e[0],e[1]) 
    newfile = open(outfile, "w") 
    wlog=csv.writer(newfile, delimiter='\t')
    
    #TimeSpan = 50.
    # origin age of taxa
    #OrigTimes = np.zeros(n_taxa) # all extant at the present
    #OrigTimes = np.random.geometric(0.3,n_taxa)-1 # geometric distrib 
    OrigTimes = np.random.uniform(0,TimeSpan,n_taxa) # uniform distribution of speciation times
    #OrigTimes[OrigTimes>45]=0 # avoid taxa appearing later than 5 Ma 
    #OrigTimes = np.random.uniform(0, TimeSpan - TimeSpan * 0.1, n_taxa)
    
    #Times = np.sort(np.random.uniform(0,50,10))  #
    Times_mod = np.sort(np.linspace(0,TimeSpan,n_bins))
    SimStates = np.zeros(n_bins) + nan
    #deltaTimes= np.diff(Times)
    #Times_mod = np.sort(np.append(Times,0))
    wlog.writerow(list(Times_mod))
    newfile.flush()

    J=0
    while J < n_taxa:
        AncState = np.random.randint(1,4,1)[0] # 0 -> 0, 1 -> A, 2 -> B, 3 -> AB
        current_state=AncState
        SimStates[0] = current_state
        t = OrigTimes[J]
        SimStates[np.nonzero(Times_mod < t)[0]] = nan
        while True:
            #print J
            if D[current_state]>0: rand_t = np.random.exponential(1/D[current_state])
            else: rand_t = inf # lineage extinct
            passed_ind = np.intersect1d(np.nonzero(Times_mod < t+rand_t)[0],np.nonzero(Times_mod > t)[0])
            #print t, rand_t
            t += rand_t
            SimStates[passed_ind] = current_state
    
            if t < max(Times_mod): # did not exceed max length
                Qrow = Q[current_state]+0
                Qrow[Qrow<0]=0
                [rate, ind] = random_choice_P(Qrow)
                current_state = ind
                #print rate, ind, current_state
                #__ if ind == 0: # lineage is extinct
                #__    break
            else: 
                log_state=list(SimStates)
                wlog.writerow(log_state)
                newfile.flush()
                J+=1
                break
    os.fsync(newfile)
    newfile.close()

 
def simulate_dataset_1area(no_sim,d,e,n_taxa,n_bins=20,wd="", area = 2):
    n_bins +=1
    def random_choice_P(vector):
        probDeath=np.cumsum(vector/np.sum(vector)) # cumulative prob (used to randomly sample one 
        r=rand.random()                          # parameter based on its deathRate)
        probDeath=np.sort(append(probDeath, r))
        ind=np.where(probDeath==r)[0][0] # just in case r==1
        return [vector[ind], ind]
    
    Q = make_Q(d,e)
    D = -np.diagonal(Q) # waiting times
    
    outfile="%s/sim_%s_%s_%s_%s_%s_%s.txt" % (wd,no_sim,n_taxa,d[0],d[1],e[0],e[1]) 
    newfile = open(outfile, "wb") 
    wlog=csv.writer(newfile, delimiter='\t')
    
    TimeSpan = 14 #50
    # origin age of taxa
    OrigTimes = np.zeros(n_taxa) # all extant at the present TH: Isn't it all species present since the beginning?
    #OrigTimes = np.random.geometric(0.3,n_taxa)-1 # geometric distrib 
    #OrigTimes = np.random.uniform(0,TimeSpan,n_taxa) # uniform distribution of speciation times
    #OrigTimes[OrigTimes>45]=0 # avoid taxa appearing later than 5 Ma 
    
    #Times = np.sort(np.random.uniform(0,50,10))  #
    Times_mod = np.sort(np.linspace(0,TimeSpan,n_bins))
    SimStates = np.zeros(n_bins) + nan
    #deltaTimes= np.diff(Times)
    #Times_mod = np.sort(np.append(Times,0))
    wlog.writerow(list(Times_mod))
    newfile.flush()

    J=0
    while J < n_taxa:
        if area == 1:
            AncState = 2 # 0 -> 0, 1 -> A, 2 -> B, 3 -> AB
        else:
            AncState = 1
        current_state=AncState
        SimStates[0] = current_state
        t = OrigTimes[J]
        SimStates[np.nonzero(Times_mod < t)[0]] = nan # Should not be needed because all species present since the 1st bin
        while True:
            #print J
            if D[current_state]>0: rand_t = np.random.exponential(1/D[current_state])
            else: rand_t = inf # lineage extinct
            passed_ind = np.intersect1d(np.nonzero(Times_mod < t+rand_t)[0],np.nonzero(Times_mod > t)[0])
            #print t, rand_t
            t += rand_t
            SimStates[passed_ind] = current_state
    
            if t < np.max(Times_mod): # did not exceed max length
                Qrow = Q[current_state]+0
                Qrow[Qrow<0]=0 # No probability for the diagonal
                [rate, ind] = random_choice_P(Qrow)
                current_state = ind
                #print rate, ind, current_state
                #__ if ind == 0: # lineage is extinct
                #__    break
            else: 
                log_state=list(SimStates)
                wlog.writerow(log_state)
                newfile.flush()
                J+=1
                break
    os.fsync(newfile)
    newfile.close()

######## PARSE INPUT FILES

def transform_Array_Tuple(A): # works ONLY for 2 areas
    A=np.array(A)
    A[np.isnan(A)]=4 # 4 means NaN
    A =np.ndarray.astype(A,dtype=int)
    z=np.empty(np.shape(A),dtype=object)
    translation = np.array([(),(0,),(1,),(0,1),(np.nan,)], dtype = object)
    z = translation[A]
    return z

def parse_input_data(input_file_name,RHO_sampling=np.ones(2),verbose=0,n_sampled_bins=0,reduce_data=0):
    try:
        DATA = np.loadtxt(input_file_name)
    except:
        tbl = np.genfromtxt(input_file_name, dtype=str, delimiter='\t')
        tbl_temp=tbl[:,1:]
        DATA=tbl_temp.astype(float)
        # remove empty taxa (absent throughout)
        ind_keep = (np.nansum(DATA,axis=1) != 0).nonzero()[0]
        DATA = DATA[ind_keep]
        if reduce_data==1: # KEEPS ONLY TAXA WITH OCCURRENCES IN BOTH AREAS
            DATA_temp = []
            if verbose == 1: print("\n\n\n",shape(DATA), "\n\n\n")
            for i in range(np.shape(DATA)[0]):
                d = DATA[i]
                d=d[np.isfinite(d)]
                areas = np.unique(d[d>0])
                if len(areas)>0:
                    if len(areas)>1 or np.max(areas)==3:
                        DATA_temp.append(DATA[i])
                        print(areas)
            DATA =  DATA_temp

    time_series = DATA[0]
    obs_area_series = transform_Array_Tuple(DATA[1:])
    nTaxa = len(obs_area_series)
    print(time_series)
    print(len(time_series),np.shape(obs_area_series))

    ###### SIMULATE SAMPLING ######
    sampled_data=list()  #np.empty(nTaxa, dtype=object)
    OrigTimeIndex=np.zeros(nTaxa)
    for l in range(nTaxa):
        obs= obs_area_series[l]
        new_obs=[]
    
        for i in range(len(obs)): 
            if i == len(obs)-1: sampling_fraction =np.ones(2) # full sampling at the present
            else: sampling_fraction = RHO_sampling
        
            if len(obs[i]) == 0: new_obs.append(()) # if no areas it means taxon is extinct for good
        
            elif np.isnan(obs[i][0]): 
                #__ if species doesn't exist yet NaN (assuming origination time is known)
                new_obs.append((np.nan,)) 
                OrigTimeIndex[l] = i+1
                #__ new_obs.append(()) # if we assume that origination time is unknown
        
            else:
                sampling_fraction = sampling_fraction[np.array(obs[i])] # RHO[0] if area (0), RHO[1] if area (1), RHO[0,1] if area (0,1)
                if verbose == 1: print(sampling_fraction, np.array(obs[i]))
                r=np.random.uniform(0,1,len(obs[i]))
                ind=np.nonzero(r<sampling_fraction)[0]
                if len(ind)>0: new_obs.append(tuple(np.array(obs[i])[ind]))    
                else: new_obs.append(())
            #print i, obs[i], new_obs[i]
    
        new_obs2,NA=[],1
        for i in range(len(new_obs)): # add NaN until first observed appearance (i.e. not true time of origin)
            if NA==1:
                if new_obs[i]==() or np.isnan(new_obs[i][0]):
                    new_obs2.append((np.nan,))
                else: 
                    new_obs2.append(new_obs[i])
                    NA = 0
                    OrigTimeIndex[l] = i
            else:
                new_obs2.append(new_obs[i])
                NA=0
        new_obs2[-1]=new_obs[-1] # present state is always sampled

        sampled_data.append(new_obs2)
        #print "TAXON" ,l,
        ##print obs
        #print new_obs2
        #_ print "ORIG",OrigTimeIndex[l], new_obs2[int(OrigTimeIndex[l])]
        
    # code data into binned time
    if n_sampled_bins > 0:
        Binned_time = np.sort(np.linspace(0,np.max(time_series),n_sampled_bins))
        #_ print Binned_time
        binned_DATA = list()
        binned_OrigTimeIndex=np.zeros(nTaxa)
        #
        for j in range(nTaxa):
            binned_states=list()
            samples_taxon_j = np.array(sampled_data[j])
            for i in range(1,n_sampled_bins):
                t1=Binned_time[i-1]
                t2=Binned_time[i]
                ind=np.intersect1d(np.nonzero(time_series>=t1)[0],np.nonzero(time_series<t2)[0])
                samples_taxon_j_time_t = samples_taxon_j[ind]
                if len(shape(samples_taxon_j_time_t)) > 1:
                    binned_states.append((nan,))
                    binned_OrigTimeIndex[j] = i
                else:
                    if (0,1) in list(samples_taxon_j_time_t): 
                        binned_states.append((0,1))
                    elif (1,) in list(samples_taxon_j_time_t) and (0,) in list(samples_taxon_j_time_t): 
                        binned_states.append((0,1))
                    elif (1,) in list(samples_taxon_j_time_t): 
                        binned_states.append((1,))
                    elif (0,) in list(samples_taxon_j_time_t): 
                        binned_states.append((0,))
                    elif () in list(samples_taxon_j_time_t): 
                        binned_states.append(())
                    else:
                        binned_states.append((nan,))
                        binned_OrigTimeIndex[j] = i

            # add present state
            binned_states.append(samples_taxon_j[-1])
            binned_DATA.append(binned_states)

        #_ print len(binned_states),len(sampled_data[0]),len(Binned_time)
            #_ 
        #_ print "\n",binned_states
        #_ 
        #_ print OrigTimeIndex.astype(int)
        #_ print binned_OrigTimeIndex.astype(int)
        binned_obs_area_series = np.array(binned_DATA)
        binned_OrigTimeIndex   = binned_OrigTimeIndex.astype(int)    
    else:
        print("Empirical bins")
        binned_obs_area_series = np.array(sampled_data, dtype = object)
        binned_OrigTimeIndex   = OrigTimeIndex.astype(int)
        Binned_time = time_series
        print(Binned_time)
                    
    return nTaxa, Binned_time, binned_obs_area_series, binned_OrigTimeIndex



def get_binned_continuous_variable(timebins, var_file):
    var = np.loadtxt(var_file,skiprows=1)
    times = var[:,0]
    values = var[:,1]
    mean_var=[]
    for i in range(1,len(timebins)):
        t_min= timebins[i-1]
        t_max= timebins[i]
        in_range_M = (times<=t_min).nonzero()[0]
        in_range_m = (times>=t_max).nonzero()[0]
        #times[np.intersect1d(in_range_M,in_range_m)]
        mean_var.append(np.mean(values[np.intersect1d(in_range_M,in_range_m)]))

    return np.array(mean_var)

def get_gamma_rates(a,YangGammaQuant,pp_gamma_ncat):
    b=a
    m = gdtrix(b,a,YangGammaQuant) # user defined categories
    s=pp_gamma_ncat/np.sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
    return np.array(m)*s # SCALED VALUES
