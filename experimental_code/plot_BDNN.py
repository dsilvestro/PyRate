import numpy as np


def get_rate_BDNN(rate, x, w): 
    # n: n species, j: traits, i: nodes
    z = np.einsum('nj,ij->ni', x, w[0], optimize=True)
    z[z < 0] = 0 
    z = np.einsum('ni,i->n', z, w[1], optimize=True)
    rates = np.exp(z) * rate
    return rates 


def get_posterior_weigths(logfile, n_traits, burnin):
    head = np.array(next(open(logfile)).split()) 
    
    w_lam_0_indx = [i for i in range(len(head)) if 'w_lam_0' in head[i]]
    w_lam_1_indx = [i for i in range(len(head)) if 'w_lam_1' in head[i]]
    
    w_mu_0_indx = [i for i in range(len(head)) if 'w_mu_0' in head[i]]
    w_mu_1_indx = [i for i in range(len(head)) if 'w_mu_1' in head[i]]
    
    post_tbl = np.loadtxt(logfile, skiprows=1)
    post_tbl = post_tbl[:int(burnin*post_tbl.shape[0]),:]
    
    nodes = len(w_lam_0_indx)/n_traits
    
    for i in range(post_tbl.shape[0]):
        w_lam_0 = post_tbl[i, w_lam_0_indx].reshape((nodes,n_traits))
        w_lam_1 = post_tbl[i, w_lam_1_indx]
    
    
    w_mu = [ post_tbl[w_mu_0_indx], post_tbl[w_mu_1_indx] ]
    
    pass


def get_posterior_rates(logfile, 
                        trait_file, 
                        time_range = np.arange(10), 
                        rescale_time = 0.015,
                        burnin = 0.2):
    
    traits = np.loadtxt(trait_file,skiprows=1)
    n_traits = traits.shape[1]
    
    if len(time_range):
        rescaled_time = rescale_time*time_range
        
        # use time as a feature
        n_traits += 1
        for i in range(len(rescaled_time)):
            time_i = rescaled_time[0]
            trait_tbl_i = 0+np.hstack((trait_tbl,rescaled_time * np.ones((trait_tbl.shape[0],1))))
        
            w_lam, w_mu = get_posterior_weigths(logfile, burnin)
    
    else:
        w_lam, w_mu = get_posterior_weigths(logfile, burnin)
    
    