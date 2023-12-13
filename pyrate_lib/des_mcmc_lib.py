import os,csv
import argparse, os,sys, time
# Prevent blas from using all CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import scipy
import scipy.linalg as linalg
from des_model_lib import *
from mcmc_lib import *
import scipy.stats
import random as rand
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import math
small_number= 1e-5


def update_positive_rate_vec(i, d):
    I=np.random.choice(len(i))
    z=np.zeros(len(i))+i
    z[I]=fabs(z[I]+(np.random.uniform(0,1,1)-.5)*d)
    return z, 0

def update_normal(i, d):
    I=np.random.choice(len(i))
    z=np.zeros(len(i))+i
    z[I]=fabs(z[I]+np.random.normal(0,d))
    return z, 0


def update_multiplier_proposal_(i,d):
    I=np.random.choice(len(i))
    z=np.zeros(len(i))+i
    u = np.random.uniform(0,1)
    l = 2*np.log(d)
    m = np.exp(l*(u-.5))
    z[I] = z[I] * m
    U=np.log(m)
    return z, U

def update_multiplier_proposal(i,d):
    S=np.shape(i)
    u = np.random.uniform(0,1,S) #*np.rint(np.random.uniform(0,f,S))
    l = 2*np.log(d)
    m = np.exp(l*(u-.5))
    #print "\n",u,m,"\n"
    ii = i * m
    U=np.sum(np.log(m))
    return ii, U

def update_multiplier_proposal_freq(q,d=1.1,f=0.75):
    S=np.shape(q)
    ff=np.random.binomial(1,f,S)
    if np.max(ff)==0:
        if len(S) == 2: 
            ff[ np.random.choice(S[0]),np.random.choice(S[1]) ] = 1
        else:
            ff[np.random.choice(S[0])] = 1
    u = np.random.uniform(0,1,S)
    l = 2*np.log(d)
    m = np.exp(l*(u-.5))
    m[ff==0] = 1.
    new_q = q * m
    U=np.sum(np.log(m))
    return new_q,U


def update_scaling_vec_V(V):
    I=np.random.choice(len(V),2,replace=False) # select 2 random time frames
    D=V[I]
    Dir=np.random.dirichlet([2,2])
    D1 = ((D*Dir)/np.sum(D*Dir))*np.sum(D)
    Z=np.zeros(len(V))+V
    Z[I] = D1
    return Z


def update_parameter_uni_2d_freq(oldL,d,f=.65,m=0,M=1):
    S=np.shape(oldL)
    ii=(np.random.uniform(0,1,S)-.5)*d
    ff=np.random.binomial(1,f,S)
    s= oldL + ii*ff
    s[s>M]=M-(s[s>M]-M)
    s[s<m]=(m-s[s<m])+m
    return s


def prior_gamma(L,a,b): return np.sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))

def prior_exp(L,rate): return np.sum(scipy.stats.expon.logpdf(L, scale=1./rate))

def prior_normal(L,loc=0,scale=1): return np.sum(scipy.stats.norm.logpdf(L,loc,scale))

def prior_beta(L, a, b): return np.sum(scipy.stats.beta.logpdf(L, a, b))

def calc_likelihood_mQ_compr(args):
    [delta_t,r_vec_list,Q_list,rho_at_present,r_vec_indexes,sign_list,sp_OrigTimeIndex,Q_index]=args
    PvDes= rho_at_present
    recursive = np.arange(sp_OrigTimeIndex,len(delta_t))[::-1]
    
    L = np.zeros((len(recursive)+1,4))
    L[0,:]=PvDes
    
    def calc_lik_bin(j,L):
        i = recursive[j]
        Q = Q_list[Q_index[i]]
        r_vec=r_vec_list[Q_index[i]]
        t=delta_t[i] 
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(abs(sign-r_vec[r_ind]),axis=1)
        # Pt = np.ones((4,4))
        # Pt= linalg.expm(Q.T *(t))
        w, vl = scipy.linalg.eig(Q,left=True, right=False)
        # w = eigenvalues
        # vl = eigenvectors
        vl_inv = np.linalg.inv(vl)
        
        
        d = np.exp(w*t)
        m1 = np.zeros((4,4))
        np.fill_diagonal(m1,d)
        Pt1 = np.dot(vl,m1)
        Pt = np.dot(Pt1,vl_inv)
        #print vl, m1, vl_inv
        
        PvDes_temp = L[j,:]
        condLik_temp= np.dot(PvDes_temp,Pt)
        PvDes= condLik_temp *rho_vec
        L[j+1,:]= PvDes
        
    [calc_lik_bin(j,L) for j in range(len(recursive))]
    
    PvDes_final=L[-1,:]
    return np.log(np.sum(PvDes_final))


def shape_sign_for_fastlik(sl, argsG, gamma_ncat):
    len_sl = len(sl)
    new_sl = []
    for i in range(len_sl):
        sli = np.array(sl[i])
        if argsG:
            s = sli.shape
            sli2 = np.zeros(s[0] * s[1] * s[2] * gamma_ncat).reshape((s[0], gamma_ncat, s[1], s[2]))
            for j in range(gamma_ncat):
                sli2[:, j, :, :] = sli
            new_sl.append(sli2)
        else:
            new_sl.append(sli)
    return new_sl


def shape_r_vec_indexes_for_fastlik(rl, argsG, gamma_ncat):
    len_rl = len(rl)
    new_rl = []
    nbins = len(rl[0])
    if argsG:
        for i in range(len_rl):
            r_vec_ind = np.array(rl[i]).flatten().reshape((nbins, 8)) # Rows: Time, cols: r_indices replicated by gamma categories
            r_vec_ind = np.tile(r_vec_ind, gamma_ncat)
            r_vec_ind = r_vec_ind.flatten() + np.repeat(np.arange(0, 4 * nbins * gamma_ncat, 4), 8)
            new_rl.append(r_vec_ind)
    else:
        for i in range(len_rl):
            r_vec_ind = np.array(rl[i]).flatten() + np.repeat(np.arange(0, 4 * nbins, 4), 8)
            new_rl.append(r_vec_ind)
    return new_rl


def precompute_Pt(delta_t, w_list, vl_list, vl_inv_list, nTaxa, traits, cat):
    recursive = np.arange(0, len(delta_t))[::-1]
    delta_t2 = np.repeat(delta_t, 4).reshape((len(delta_t), 4))
    if traits or cat:
        delta_t2 = np.tile(delta_t2.T, nTaxa).T
        len_rec = len(recursive)
        recursive = np.tile(recursive, nTaxa) + np.repeat(np.arange(0, len_rec * nTaxa, len_rec), len_rec)
    d = np.exp(w_list * delta_t2)
    m1 = np.zeros((len(recursive), 4, 4))
    np.einsum('ijj->ij', m1)[...] = d # Fill diagonal of a 3D array
    Pt1 = np.matmul(vl_list, m1)
    Pt = np.matmul(Pt1, vl_inv_list)
    Pt = Pt[recursive, :]
    return Pt


#def get_eigen_list(Q_list):
#    L=len(Q_list)
#    w_list,vl_list,vl_inv_list = [],[],[]
#    for Q in Q_list:
#        w, vl = scipy.linalg.eig(Q,left=True, right=False) # w = eigenvalues; vl = eigenvectors
#        vl_inv = np.linalg.inv(vl)
#        w_list.append(w)
#        vl_list.append(vl)
#        vl_inv_list.append(vl_inv)
#    return w_list,vl_list,vl_inv_list


def get_eigen_list(QT_array):
    # Requires 3D array with transposed Q matrices along axis 0!
    w, vl = np.linalg.eig(QT_array)
    vl = vl[:,:,[1,2,3,0]] * [-1,1,-1,1]
    w = w[:,[1,2,3,0]]
    vl_inv = np.linalg.inv(vl)
    return w, vl, vl_inv


# If numba is installed, use compiled matrix product (20% faster likelihood calculation than with numpy)
try:
    import numba as nb
    @nb.njit(fastmath = True, parallel = False)
    def calc_lik_bin(L, len_rec, Pt_taxon, rho_vec2):
        for j in range(len_rec):
            for col_ind in range(4):
                for k in range(4):
                    L[j + 1, col_ind] += L[j, k] * Pt_taxon[j, k, col_ind]
                L[j + 1, col_ind] = L[j + 1, col_ind] * rho_vec2[j, col_ind]
        return L

    @nb.njit(fastmath = True, parallel = False) #['(float64[:,:,:], int64, float64[:,:,:], float64[:,:])']
    def calc_lik_bin_gamma(L, len_rec, Pt_taxon, rho_vec2, ncat):
        for j in range(len_rec):
            for row_ind in range(ncat):
                for col_ind in range(4):
                    for k in range(4):
                        L[j + 1, row_ind, col_ind] += L[j, row_ind, k] * Pt_taxon[j, k, col_ind]
                    L[j + 1, row_ind, col_ind] = L[j + 1, row_ind, col_ind] * rho_vec2[j, row_ind, col_ind]
        return L

except:
    def calc_lik_bin(L, len_rec, Pt_taxon, rho_vec2):
        for j in range(len_rec):
            L[j + 1] = (L[j] @ Pt_taxon[j, :, :]) * rho_vec2[j, :]
        return L

    def calc_lik_bin_gamma(L, len_rec, Pt_taxon, rho_vec2, ncat):
        for j in range(len_rec):
            #PvDes_temp = L[j]
            #condLik_temp = PvDes_temp @ Pt_taxon[j, :, :]
            #PvDes = condLik_temp * rho_vec2[j, :]
            L[j + 1] = (L[j] @ Pt_taxon[j, :, :]) * rho_vec2[j, :]
        return L


def calc_likelihood_mQ_eigen_precompute(args):
    [r_vec, rho_at_present, r_vec_indexes, sign_list, recursive, Pt_taxon] = args
    len_rec = len(recursive)
    nbins = len(sign_list)
    rho_gamma = r_vec[0].ndim > 1
    m = r_vec.flatten()[r_vec_indexes]
    if rho_gamma:
        gamma_ncat = r_vec[0].shape[0]
        m = m.reshape((nbins, gamma_ncat, 4, 2))
    else:
        m = m.reshape((nbins, 4, 2))

    rho_vec2 = np.prod(np.abs(sign_list[recursive] - m[recursive]), axis = -1)

    if rho_gamma:
        L = np.zeros((len_rec + 1, gamma_ncat, 4))
        L[0, :, :] = np.tile(rho_at_present, gamma_ncat).reshape((gamma_ncat, 4))
        L = calc_lik_bin_gamma(L, len_rec, Pt_taxon, rho_vec2, gamma_ncat)
    else:
        L = np.zeros((len_rec + 1, 4))
        L[0, :] = rho_at_present
        L = calc_lik_bin(L, len_rec, Pt_taxon, rho_vec2)
    
    lik_mQ = np.sum(L[-1], axis = -1)
    if np.any(lik_mQ > 0):
        return lik_mQ
    else: 
        return -np.inf


def calc_likelihood_mQ_eigen(args):
    [delta_t,r_vec_list,w_list,vl_list,vl_inv_list,rho_at_present,r_vec_indexes,sign_list,recursive,index_r,index_q]=args
    PvDes= rho_at_present
    L = np.zeros((len(recursive)+1,4))
    L[0,:]=PvDes
    
    def calc_lik_bin(j,L):
        i = recursive[j]
        ind_Q = index_q[i]
        r_vec=r_vec_list[i]#r_vec_list[index_r[i]]
        t=delta_t[i] 
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(np.abs(sign-r_vec[r_ind]),axis=1)
        d= np.exp(w_list[ind_Q,:]*t)
        m1 = np.zeros((4,4))
        np.fill_diagonal(m1,d)
        Pt1 = np.dot(vl_list[ind_Q,:],m1)
        Pt = np.dot(Pt1,vl_inv_list[ind_Q,:])
        PvDes_temp = L[j,:]
        condLik_temp= np.dot(PvDes_temp,Pt)
        PvDes= condLik_temp *rho_vec
        L[j+1,:]= PvDes
        
    [calc_lik_bin(j,L) for j in range(len(recursive))]    
    PvDes_final=L[-1,:]
    if np.sum(PvDes_final) <= 0: 
        #print np.sum(PvDes_final), list(PvDes_final)
        return -np.inf
    else: 
        return np.sum(PvDes_final)


def calc_likelihood_mQ_eigen_aprx(args):
    [delta_t,r_vec_list,w_list,vl_list,vl_inv_list,rho_at_present,r_vec_indexes,sign_list,sp_OrigTimeIndex,Q_index]=args
    PvDes= rho_at_present
    recursive = np.arange(sp_OrigTimeIndex,len(delta_t))[::-1]
    L = np.zeros((len(recursive)+1,4))
    L[0,:]=PvDes
    print(recursive, len(recursive))
    
    def calc_lik_bin(j,L,t=0):
        i = recursive[j]
        ind_Q = Q_index[i]
        r_vec=r_vec_list[Q_index[i]]
        if t==0: t=delta_t[i] 
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(np.abs(sign-r_vec[r_ind]),axis=1)
        d= np.exp(w_list[ind_Q]*t)
        m1 = np.zeros((4,4))
        np.fill_diagonal(m1,d)
        Pt1 = np.dot(vl_list[ind_Q],m1)
        Pt = np.dot(Pt1,vl_inv_list[ind_Q])
        PvDes_temp = L[j,:]
        condLik_temp= np.dot(PvDes_temp,Pt)
        PvDes= condLik_temp *rho_vec
        print(j, rho_vec, list(PvDes), r_vec_list, i) # d.astype(float)
        L[j+1,:]= PvDes
        return PvDes
        
    def calc_lik_bin_mod(j,L,t=0):
        i = recursive[j]
        ind_Q = Q_index[i]
        r_vec=r_vec_list[Q_index[i]]
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(abs(sign-r_vec[r_ind]),axis=1)
        print(sign, r_vec[r_ind], rho_vec)
        d= np.exp(w_list[ind_Q]*t)
        m1 = np.zeros((4,4))
        np.fill_diagonal(m1,d)
        Pt1 = np.dot(vl_list[ind_Q],m1)
        Pt = np.dot(Pt1,vl_inv_list[ind_Q])
        PvDes_temp = L[j,:]
        condLik_temp= np.dot(PvDes_temp,Pt)
        s_prob = np.log(np.exp(.25*4))
        PvDes= condLik_temp * np.array([0,0,0,s_prob**2])
        print(j, rho_vec, list(PvDes), r_vec_list, i) # d.astype(float)
        L[j+1,:]= PvDes
        #quit()
        return PvDes
        
    
    
    
    print(calc_lik_bin_mod(0,L,t=8))
    
    
    [calc_lik_bin(j,L) for j in range(len(recursive))]    
    PvDes_final=L[-1,:]
    quit()
    
    #print np.log(np.sum(PvDes_final))
    return np.log(np.sum(PvDes_final))



def calc_likelihood_mQ(args):
    [delta_t,r_vec_list,Q_list,rho_at_present,r_vec_indexes,sign_list,recursive,index_r,index_q]=args
    PvDes= rho_at_present
    #print rho_at_present
    for i in recursive:
        #print "here",i, Q_index[i]
        Q = Q_list[index_q[i],:]
        r_vec=r_vec_list[i]#r_vec_list[index_r[i]]
        # get time span
        t=delta_t[i] 
        # get rho vector
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(np.abs(sign-r_vec[r_ind]), axis=1)
        # prob of at least 1  1-exp(-rho_vec*t)
        Pt= linalg.expm(Q *(t))
        condLik_temp= np.dot(PvDes,Pt)
        PvDes= condLik_temp *rho_vec
        
        #print "temp,",t,PvDes,log(condLik_temp),rho_vec
    if np.sum(PvDes) <= 0: 
        print(np.sum(PvDes), list(PvDes))
    
    return np.sum(PvDes)






def calc_likelihood(args):
    [delta_t,r_vec,Q,rho_at_present,r_vec_indexes,sign_list,sp_OrigTimeIndex]=args
    PvDes= rho_at_present
    #print rho_at_present
    recursive = np.arange(sp_OrigTimeIndex,len(delta_t))[::-1]
    for i in recursive:
        # get time span
        t=delta_t[i] 
        # get rho vector
        r_ind= r_vec_indexes[i]
        sign=  sign_list[i]
        rho_vec= np.prod(np.abs(sign-r_vec[r_ind]), axis=1)
        # prob of at least 1  1-exp(-rho_vec*t)
        #print rho_vec,r_vec[r_ind]
        #print obs_area_series[i],"\t",rho_vec                
        Pt= linalg.expm(Q.T *(t))
        condLik_temp= np.dot(PvDes,Pt)
        PvDes= condLik_temp *rho_vec
        
        #print "temp,",t,PvDes,log(condLik_temp),rho_vec
    return np.log(np.sum(PvDes))


def gibbs_sampler_hp(x,hp_alpha,hp_beta):
    g_shape=hp_alpha+len(x)
    rate=hp_beta+np.sum(x)
    lam = np.random.gamma(shape= g_shape, scale= 1./rate)
    return lam


def get_temp_TI(k=10,a=0.3):
    K=k-1.        # K+1 categories
    k=array(np.arange(int(K+1)))
    beta=k/K
    alpha=a            # categories are beta distributed
    temperatures=beta**(1./alpha)
    temperatures[0]+= small_number # avoid exactly 0 temp
    temperatures=temperatures[::-1]
    return temperatures

