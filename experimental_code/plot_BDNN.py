import numpy as np
np.set_printoptions(suppress= 1, precision=3)
import os, csv, sys
import pandas as pd


def softPlus(z):
    return np.log(np.exp(z) + 1)

def expFun(z):
    return np.exp(z)

def get_rate_BDNN(rate, x, w, outputfun=0): 
    actfun = [softPlus, expFun][outputfun]
    # n: n species, j: traits, i: nodes
    # print(x.shape, w[0].shape)
    z = np.einsum('nj,ij->ni', x, w[0])
    z[z < 0] = 0 
    z = np.einsum('ni,i->n', z, w[1])
    rates = actfun(z) * rate
    return rates 


def get_posterior_weigths(logfile, n_traits, burnin):
    head = np.array(next(open(logfile)).split()) 
    
    w_lam_0_indx = [i for i in range(len(head)) if 'w_lam_0' in head[i]]
    w_lam_1_indx = [i for i in range(len(head)) if 'w_lam_1' in head[i]]
    rate_l_0_indx = [i for i in range(len(head)) if head[i] == 'lambda_0']
    
    w_mu_0_indx = [i for i in range(len(head)) if 'w_mu_0' in head[i]]
    w_mu_1_indx = [i for i in range(len(head)) if 'w_mu_1' in head[i]]
    rate_m_0_indx = [i for i in range(len(head)) if head[i] == 'mu_0']
    
    post_tbl = np.loadtxt(logfile, skiprows=1)
    post_tbl = post_tbl[int(burnin*post_tbl.shape[0]):,:]
    
    nodes = int(len(w_lam_0_indx)/n_traits)
    w_lam_list = []
    rate_l = post_tbl[:,rate_l_0_indx]
    for i in range(post_tbl.shape[0]):
        w_lam_0 = post_tbl[i, w_lam_0_indx].reshape((nodes,n_traits))
        w_lam_1 = post_tbl[i, w_lam_1_indx]
        w_lam_list.append([w_lam_0,w_lam_1])
    
    w_mu_list = []
    rate_m = post_tbl[:,rate_m_0_indx]
    for i in range(post_tbl.shape[0]):
        w_mu_0 = post_tbl[i, w_mu_0_indx].reshape((nodes,n_traits))
        w_mu_1 = post_tbl[i, w_mu_1_indx]
        w_mu_list.append([w_mu_0,w_mu_1])
        
    return rate_l, rate_m, w_lam_list, w_mu_list


def get_posterior_weigths_BDS(logfile, n_traits, burnin):
    head = np.array(next(open(logfile)).split()) 
    
    tot_length_indx = [i for i in range(len(head)) if head[i] == 'tot_length'][0]
    w_lam_0_indx = [i for i in range(len(head)) if 'w_lam_0' in head[i]]
    w_lam_1_indx = [i for i in range(len(head)) if 'w_lam_1' in head[i]]
    rate_l_0_indx = [i for i in range(len(head)) if 'lambda_' in head[i] and i < tot_length_indx]
    
    w_mu_0_indx = [i for i in range(len(head)) if 'w_mu_0' in head[i]]
    w_mu_1_indx = [i for i in range(len(head)) if 'w_mu_1' in head[i]]
    rate_m_0_indx = [i for i in range(len(head)) if 'mu_' in head[i] and i < tot_length_indx]
    
    post_tbl = np.loadtxt(logfile, skiprows=1)
    post_tbl = post_tbl[int(burnin*post_tbl.shape[0]):,:]
    
    nodes = int(len(w_lam_0_indx)/n_traits)
    w_lam_list = []
    for i in range(post_tbl.shape[0]):
        w_lam_0 = post_tbl[i, w_lam_0_indx].reshape((nodes,n_traits))
        w_lam_1 = post_tbl[i, w_lam_1_indx]
        w_lam_list.append([w_lam_0,w_lam_1])
        
    w_mu_list = []
    rate_m = post_tbl[:,rate_m_0_indx]
    for i in range(post_tbl.shape[0]):
        w_mu_0 = post_tbl[i, w_mu_0_indx].reshape((nodes,n_traits))
        w_mu_1 = post_tbl[i, w_mu_1_indx]
        w_mu_list.append([w_mu_0,w_mu_1])
    
    rate_l_list = post_tbl[:,rate_l_0_indx]
    rate_m_list = post_tbl[:,rate_m_0_indx]
        
    return rate_l_list, rate_m_list, w_lam_list, w_mu_list

def get_file_name(s):
    input_file_raw = os.path.basename(s)
    input_file = os.path.splitext(input_file_raw)[0]  # file name without extension
    return input_file
    

def get_posterior_rates(logfile, 
                        trait_file, 
                        time_range = np.arange(15), 
                        rescale_time = 0.015,
                        burnin = 0.25):
    
    """
    traits_raw = np.genfromtxt(trait_file,skip_header=1, dtype=str)
    traits = traits_raw[:,1:].astype(float)
    n_traits = traits.shape[1]
    # use time as a feature
    n_traits += 1
    
    a = np.min(traits, axis=0)
    b = np.max(traits, axis=0)
    c = np.median(traits, axis=0)
    
    trait_select = np.array([
                             [a[0],c[1],0], # low Lat Woody
                             [c[0],c[1],0], # mid Lat Woody
                             [b[0],c[1],0], # high Lat Woody
                             [a[0],c[1],1],
                             [c[0],c[1],1],
                             [b[0],c[1],1]
                         
                         
                         ])

    rate_l, rate_m, w_lam, w_mu = get_posterior_weigths(logfile, n_traits, burnin)
    
    if len(time_range):
        rescaled_time = rescale_time*time_range
    
    
    for i in range(len(rescaled_time)):
        time_i = rescaled_time[i]
        trait_tbl_i = 0+np.hstack((trait_select,time_i * np.ones((trait_select.shape[0],1))))
            
            
        lam_matrix = np.zeros((len(rate_l),trait_select.shape[0]))
        mu_matrix = np.zeros((len(rate_l),trait_select.shape[0]))
        
        for j in range(len(rate_l)):
            vec_lam_i = get_rate_BDNN(rate_l[j], trait_tbl_i, w_lam[j])
            vec_mu_i = get_rate_BDNN(rate_m[j], trait_tbl_i, w_mu[j])
            lam_matrix[j,:] = vec_lam_i
            mu_matrix[j,:] = vec_mu_i
            
                
                
                
        print("\ntime", time_i/rescale_time)
        print("lambda:",np.mean(lam_matrix, axis=0))
        print("mu:",np.mean(mu_matrix, axis=0))
    """
    pass

def predicted_rates(logfile, 
                    trait_file, 
                    time_range = np.arange(15), 
                    rescale_time = 0.015,
                    burnin = 0.25,
                    fixShift = [np.inf,56.0,33.9,23.03,5.333,2.58,0],
                    time_as_trait = True):
    
    traits = np.loadtxt(trait_file, skiprows=1)
    n_traits = traits.shape[1]
    # use time as a feature
    if time_as_trait:   
        n_traits += 1
    
    rate_l2D, rate_m2D, w_lam, w_mu = get_posterior_weigths_BDS(logfile, n_traits, burnin)
    
    rescaled_time = rescale_time*time_range
    
    for i in range(len(rescaled_time)):
        time_i = rescaled_time[i]
        if time_as_trait:
            trait_tbl_i = 0+np.hstack((traits,time_i * np.ones((traits.shape[0],1))))
        else:
            trait_tbl_i = 0+traits
        
        rate_l = rate_l2D[:, np.digitize(time_range[i], fixShift)-1] 
        rate_m = rate_m2D[:, np.digitize(time_range[i], fixShift)-1] 
        # print(np.mean(rate_l), np.mean(rate_m))
            
        lam_matrix = np.zeros((len(rate_l),traits.shape[0]))
        mu_matrix = np.zeros((len(rate_l),traits.shape[0]))
        
        for j in range(len(rate_l)):
            vec_lam_i = get_rate_BDNN(rate_l[j], trait_tbl_i, w_lam[j])
            vec_mu_i = get_rate_BDNN(rate_m[j], trait_tbl_i, w_mu[j])
            lam_matrix[j,:] = vec_lam_i
            mu_matrix[j,:] = vec_mu_i
            
        # get harmonic mean of rates
        # print(lam_matrix.shape)
        lam_matrix_hm = np.mean(lam_matrix,axis=0) #len(rate_l) / np.sum(1/lam_matrix,axis=0)
        mu_matrix_hm =  np.mean(mu_matrix,axis=0) #len(rate_l) / np.sum(1/mu_matrix,axis=0)
        print(time_range[i], "MAX",np.max(lam_matrix_hm), np.median(mu_matrix_hm), lam_matrix_hm.shape)
        
        rates_predicted = np.array([lam_matrix_hm, mu_matrix_hm]).T

        out_file_l = get_file_name(trait_file) + "_t%s_NN%s.txt" % (time_range[i], w_lam[0][0].shape[0]) 
        np.savetxt(os.path.join(os.path.dirname(trait_file), out_file_l),rates_predicted, delimiter="\t")
        # out_file_m = get_file_name(trait_file) + "_mu_NN%s.txt" % w_lam[0][0].shape[0]
        # np.savetxt(os.path.join(os.path.dirname(trait_file), out_file_m),lam_matrix_hm, delimiter="\t")

def get_tste_from_logfile(f, burnin=0):
    head = next(open(f)).split()
    t_file=np.loadtxt(f, skiprows=1)
    w=[x for x in head if 'TS' in x]
    #w=[x for x in head if 'ts_' in x]
    ind_ts0 = head.index(w[0])
    y=[x for x in head if 'TE' in x]
    #y=[x for x in head if 'te_' in x]
    ind_te0 = head.index(y[0])
    print(len(w), "species", t_file.shape)
    j=0
    out_list=list()
    if burnin<1: burnin = int(burnin*t_file.shape[0])
    
    out_list = []
    for i in np.arange(ind_ts0,ind_te0):
        meanTS= np.mean(t_file[burnin:t_file.shape[0],i])
        meanTE= np.mean(t_file[burnin:t_file.shape[0],ind_te0+j])
        out_list.append([meanTS, meanTE])
        j += 1
    
    species_list = [s.split("_TS")[0] for s in w]
    return out_list, species_list
    

def predict_rates_per_species(logfile, 
                              species_trait_file=None,
                              trait_tbl=None,
                              wd="", 
                              time_range = np.arange(15), 
                              rescale_time = 0.015,
                              burnin = 0.25,
                              fixShift = [np.inf,56.0,33.9,23.03,5.333,2.58,0],
                              return_post_sample=False,
                              out=""):
    
    if species_trait_file:
        species_traits = pd.read_csv(species_trait_file, delimiter="\t")
        traits = species_traits.iloc[:,1:]
    elif trait_tbl is not None:
        species_traits = trait_tbl
        traits = species_traits.iloc[:,1:]
        n_traits = traits.shape[1]
        n_taxa = traits.shape[0]
    else:
        # sys.exit("No traits found")
        species_traits = pd.DataFrame([["species"]])
        species_traits.columns=['Taxon_name']
        traits = None
        n_taxa = 1
        n_traits = 0
    # use time as a feature
    
    if rescale_time > 0:
        rescaled_time = rescale_time*time_range
        time_as_trait = True
        n_traits += 1
    elif rescale_time == 0:
        rescaled_time = 1*time_range
        time_as_trait = False        
    else:
        sys.exit("Option not available")

    rate_l2D, rate_m2D, w_lam, w_mu = get_posterior_weigths_BDS(logfile, n_traits, burnin)
    print("N. traits: ", n_traits, rate_l2D.shape)
    
    
    species_rate_lam = []
    species_rate_mu  = []
    species_rate_div = []
    
    rate_samples = list()
    
    for i in range(len(rescaled_time)):
        time_i = rescaled_time[i]
        if time_as_trait:
            if traits is not None:
                trait_tbl_i = 0 + np.hstack((traits,time_i * np.ones((n_taxa,1))))
            else:
                trait_tbl_i = 0 + time_i * np.ones((n_taxa,1))
        else:
            trait_tbl_i = 0 + traits
        
        rate_l = rate_l2D[:, np.digitize(time_range[i], fixShift)-1] 
        # print('rate_l', rate_l)
        rate_m = rate_m2D[:, np.digitize(time_range[i], fixShift)-1] 
        # print(np.mean(rate_l), np.mean(rate_m))
            
        lam_matrix = np.zeros((len(rate_l),n_taxa))
        mu_matrix = np.zeros((len(rate_l),n_taxa))
        
        for j in range(len(rate_l)):
            vec_lam_i = get_rate_BDNN(rate_l[j], trait_tbl_i, w_lam[j])
            vec_mu_i = get_rate_BDNN(rate_m[j], trait_tbl_i, w_mu[j])
            lam_matrix[j,:] = vec_lam_i
            mu_matrix[j,:] = vec_mu_i
            
        # get harmonic mean of rates
        # print(lam_matrix.shape)
        lam_matrix_hm =  len(rate_l) / np.sum(1/lam_matrix,axis=0) # np.mean(lam_matrix,axis=0)
        mu_matrix_hm =   len(rate_l) / np.sum(1/mu_matrix,axis=0) # np.mean(mu_matrix,axis=0) 
        print(time_range[i], "rates",np.median(lam_matrix_hm), np.median(mu_matrix_hm), lam_matrix_hm.shape)
        net_div = lam_matrix - mu_matrix
        div_matrix_hm =  np.mean(net_div,axis=0) 
        
        if return_post_sample:
            res = [lam_matrix, mu_matrix]
            rate_samples.append(res)
        
        species_rate_lam.append(lam_matrix_hm)
        species_rate_mu.append(mu_matrix_hm)
        species_rate_div.append(div_matrix_hm)
    
    species_rate_lam = np.array(species_rate_lam).T
    species_rate_mu = np.array(species_rate_mu).T
    species_rate_div = np.array(species_rate_div).T
    
    list_tste, species_list_in_log_file = get_tste_from_logfile(logfile, burnin)    
    # species_list_in_log_file might be different from list in trait file
    species_list_in_log_file = np.array(species_list_in_log_file)
    with open(os.path.join(wd, "taxon_speciation_rates%s.txt" % out), 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    l = ["Species","ts","te"]
                    h = ["%s_Ma" % time_range[i] for i in  range(len(rescaled_time))]
                    writer.writerow(l+h) 
                    # print(species_list_in_log_file)
                    for i in range(len(species_rate_lam)):
                        if species_traits["Taxon_name"][i] in species_list_in_log_file:
                            indx = np.where(species_list_in_log_file == species_traits["Taxon_name"][i])[0][0]
                            # print(indx ,species_traits["Taxon_name"][i])
                            l = [species_traits["Taxon_name"][i]] + list_tste[indx] + list(species_rate_lam[i])
                        else:
                            l = [species_traits["Taxon_name"][i]] + ['NA', 'NA'] + list(species_rate_lam[i])
                        writer.writerow(l) 

    with open(os.path.join(wd, "taxon_extinction_rates%s.txt" % out), 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    l = ["Species","ts","te"]
                    h = ["%s_Ma" % time_range[i] for i in  range(len(rescaled_time))]
                    writer.writerow(l+h) 
                    for i in range(len(species_rate_mu)):
                        if species_traits["Taxon_name"][i] in species_list_in_log_file:
                            indx = np.where(species_list_in_log_file == species_traits["Taxon_name"][i])[0][0]
                            l = [species_traits["Taxon_name"][i]] + list_tste[indx] + list(species_rate_mu[i])
                        else:
                            l = [species_traits["Taxon_name"][i]] + ['NA', 'NA'] + list(species_rate_mu[i])
                        writer.writerow(l) 
    
    with open(os.path.join(wd, "taxon_diversification_rates%s.txt" % out), 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    l = ["Species","ts","te"]
                    h = ["%s_Ma" % time_range[i] for i in  range(len(rescaled_time))]
                    writer.writerow(l+h) 
                    for i in range(len(species_rate_div)):
                        if species_traits["Taxon_name"][i] in species_list_in_log_file:
                            indx = np.where(species_list_in_log_file == species_traits["Taxon_name"][i])[0][0]
                            l = [species_traits["Taxon_name"][i]] + list_tste[indx] + list(species_rate_div[i])
                        else:
                            l = [species_traits["Taxon_name"][i]] + ['NA', 'NA'] + list(species_rate_div[i])
                        writer.writerow(l) 
    
    print("\nSummary tables with species-specific rates saved in: ")
    print(wd)
    print("as:\ntaxon_speciation_rates%s.txt \ntaxon_extinction_rates%s.txt \ntaxon_diversification_rates%s.txt" % \
    (out, out, out))

    if return_post_sample:
        return np.array(rate_samples)


def parse_sum_txt_file(fname):
    # fname = "/Users/dsilvestro/Documents/Projects/Ongoing/Fer_Juan/bdnn_analysis/pyrate_mcmc_logs/Europe_Sites_1_m2_G_BDS_BDNN3_sum.txt"
    # fname = "/Users/dsilvestro/Documents/Projects/Ongoing/Fer_Juan/bdnn_analysis/pyrate_mcmc_logs/Europe_Sites_1_m3_G_BDS_BDNN3T_sum.txt"
    with open(fname) as file:
        lines = file.readlines()    
    
    for l in lines:
        if "fixed times of rate shift: " in l:
            t_shift = l.split("fixed times of rate shift: ")[1]
            t_shift = t_shift.split(" ")
            t_shift = np.array([np.inf] + t_shift[:-1]).astype(float)
        if "BDNNtimetrait" in l:
            time_as_trait_tmp = l.split("BDNNtimetrait=")[1]
            time_as_trait = float(time_as_trait_tmp.split(",")[0])
    
    mid_points = -np.diff(t_shift[1:])/2 + t_shift[2:]
    return t_shift, mid_points, time_as_trait




if __name__ == '__main__': 
    
    species_trait_file= 'NTemFloraTraits.txt'    
    logfile = 'NTemFlora_1_BDS_BDNN7T_mcmc.log'
    wd = 'logs/'
    time_range = np.arange(15)
    rescale_time = 0.015
    burnin = 0.75
    time_as_trait = True
    fixShift = [np.inf,56.0,47.8,41.2,37.8,33.9,28.1,23.03,20.44,15.97,13.82,11.63,7.246,5.333,3.6,2.58]

    predict_rates_per_species(logfile, 
                                species_trait_file,
                                wd, 
                                time_range = np.arange(0, 65, 1), 
                                rescale_time = 0.015,
                                burnin = 0.75,
                                fixShift = [np.inf,56.0,33.9,23.03,5.333,2.58,0],
                                time_as_trait = True)
    
 