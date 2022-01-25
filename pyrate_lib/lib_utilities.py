#!/usr/bin/env python 
import argparse, os,sys, platform
from numpy import *
import numpy as np
import os, csv, glob
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3)   # rounds all array elements to 3rd digit
import collections
import itertools
from scipy import stats
import pyrate_lib.lib_DD_likelihood as lib_DD_likelihood
self_path=os.getcwd()

def rescale_vec_to_range(x, r=1., m=0):
        print("we rescale")
        temp = (x-min(x))/(max(x)-min(x))
        temp = temp*r # rescale
        temp = temp+m # shift
        return(temp)

def calcHPD(data, level=0.95) :
    assert (0 < level < 1)    
    d = list(data)
    d.sort()    
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        raise RuntimeError("not enough data")    
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k
    assert 0 <= i <= i+nIn-1 < len(d)    
    return np.array([d[i], d[i+nIn-1]])


def print_R_vec(name,v):
    new_v=[]
    if len(v)==0: vec= "%s=c()" % (name)
    elif len(v)==1: vec= "%s=c(%s)" % (name,v[0])
    elif len(v)==2: vec= "%s=c(%s,%s)" % (name,v[0],v[1])
    else:
        for j in range(0,len(v)): 
            value=v[j]
            if isnan(v[j]): value="NA"
            new_v.append(value)
        
        vec="%s=c(%s, " % (name,new_v[0])
        for j in range(1,len(v)-1): vec += "%s," % (new_v[j])
        vec += "%s)"  % (new_v[j+1])
    return vec


def write_ts_te_table(path_dir, tag="",clade=0,burnin=0.1,plot_ltt=True, n_samples=1):
    if clade== -1: clade=0 # clade set to -1 by default
    
    direct="%s/*%s*mcmc.log" % (path_dir,tag)
    files=glob.glob(direct)
    files=sort(files)
    if len(files)==0:
        files=[path_dir]
        path_dir = os.path.dirname(path_dir)
        if path_dir=="": path_dir= self_path
        
    print("found", len(files), "log files...\n")
    print(files)
    print(tag)
    count=0

    if len(files)==1 or tag != "":
        name_file = os.path.splitext(os.path.basename(files[0]))[0]
        name_file = name_file.split("_mcmc")[0]    
        outfile="%s/%s_se_est.txt" % (path_dir, name_file)
        newfile = open(outfile, "w") 
        wlog=csv.writer(newfile, delimiter='\t')
        head="clade\tspecies"+ ("\tts\tte"*(len(files)*n_samples))
        wlog.writerow(head.split('\t'))
        newfile.flush()

    for f in files:
        if 2>1: #try:
            #t_file=np.genfromtxt(f, delimiter='\t', dtype=None)
            t_file=np.loadtxt(f, skiprows=1)
            input_file = os.path.basename(f)
            name_file = os.path.splitext(input_file)[0]
            path_dir = "%s/" % os.path.dirname(f)
            wd = "%s" % os.path.dirname(f)
            shape_f=list(shape(t_file))
            print("%s" % (name_file), end=' ')
            
            
            if len(files)>1 and tag=="":
                name_file1 = name_file.split("_mcmc")[0]    
                outfile="%s/%s_se_est.txt" % (path_dir, name_file1)
                newfile = open(outfile, "w") 
                wlog=csv.writer(newfile, delimiter='\t')
                head="clade\tspecies"+ ("\tts\tte"*(n_samples))
                wlog.writerow(head.split('\t'))
                newfile.flush()
                
            
            
            #if count==0:
            head = next(open(f)).split()
            w=[x for x in head if 'TS' in x]
            #w=[x for x in head if 'ts_' in x]
            ind_ts0 = head.index(w[0])
            y=[x for x in head if 'TE' in x]
            #y=[x for x in head if 'te_' in x]
            ind_te0 = head.index(y[0])
            print(len(w), "species", shape_f)
            j=0
            out_list=list()
            if burnin<1: burnin = int(burnin*shape_f[0])
            
            if n_samples == 1:
                for i in arange(ind_ts0,ind_te0):
                    meanTS= np.mean(t_file[burnin:shape_f[0],i].astype(float))
                    meanTE= np.mean(t_file[burnin:shape_f[0],ind_te0+j].astype(float))
                    j+=1
                    if count==0: out_list.append(array([clade, j, meanTS, meanTE]))
                    else: out_list.append(array([meanTS, meanTE]))
            
            else:
                for i in arange(ind_ts0,ind_te0):
                    indx = np.random.choice(range(burnin, shape_f[0]-1),n_samples)
                    TS= list(t_file[burnin:shape_f[0],i].astype(float)[indx])
                    TE= list(t_file[burnin:shape_f[0],ind_te0+j].astype(float)[indx])
                    j+=1
                    #print(indx, n_samples, TS)
                    
                    listTSTE = list(np.array([y for y in zip(TS,TE)] ).flatten())
                    if count==0: 
                        out_list.append(array([clade, j] + listTSTE))
                    else: out_list.append(array(listTSTE))
        
                    #print i-ind_ts0, array([meanTS,meanTE])
            
            out_list=array(out_list)
            #print(out_list)
            
            if plot_ltt is True and count==0:                
                ### plot lineages and LTT
                ts = out_list[:,2+count]
                te = out_list[:,3+count]
                print(np.shape(ts))
                title = name_file
                time_events=sort(np.unique(np.concatenate((ts,te),axis=0)))[::-1]
                div_trajectory = lib_DD_likelihood.get_DT(time_events,ts,te)

                # R - plot lineages
                out="%s/%s_LTT.r" % (wd,name_file)
                r_file = open(out, "w") 
    
                if platform.system() == "Windows" or platform.system() == "Microsoft":
                    wd_forward = os.path.abspath(wd).replace('\\', '/')
                    r_script= "\n\npdf(file='%s/%s_LTT.pdf',width=0.6*20, height=0.6*10)\n" % (wd_forward,name_file)
                else: 
                    r_script= "\n\npdf(file='%s/%s_LTT.pdf',width=0.6*20, height=0.6*10)\n" % (wd,name_file)
    
                R_ts = print_R_vec("ts",ts)
                R_te = print_R_vec("te",te)
                R_div_trajectory = print_R_vec("div_traj",div_trajectory)
                R_time_events    = print_R_vec("time_events",time_events)

                r_script += """title = "%s"\n%s\n%s\n%s\n%s""" % (name_file,R_ts,R_te,R_div_trajectory,R_time_events)

                r_script += """
                par(mfrow=c(1,2))
                L = length(ts)
                plot(ts, 1:L , xlim=c(-max(ts)-1,0), pch=20, type="n", main=title,xlab="Time (Ma)",ylab="Lineages")
                for (i in 1:L){segments(x0=-te[i],y0=i,x1=-ts[i],y1=i)}    
                t = -time_events
                plot(div_traj ~ t,type="s", main = "Diversity trajectory",xlab="Time (Ma)",ylab="Number of lineages",xlim=c(-max(ts)-1,0))
                abline(v=c(65,200,251,367,445),lty=2,col="gray")
                """
                
                r_file.writelines(r_script)
                r_file.close()
                print("\nAn LTT plot was saved as: %sLTT.pdf" % (name_file))
                print("\nThe R script with the source for the LTT plot was saved as: %sLTT.r\n(in %s)" % (name_file, wd))
                if platform.system() == "Windows" or platform.system() == "Microsoft":
                    cmd="cd %s & Rscript %s_LTT.r" % (wd,name_file)
                else: 
                    cmd="cd %s; Rscript %s/%s_LTT.r" % (wd,wd,name_file)
                os.system(cmd)
                print("done\n")
                
                ### end plot lineages and LTT
                
            
                
            if count==0: out_array=out_list
            else: out_array=np.hstack((out_array, out_list))
            
            if len(files)>1 and tag=="":
                print(shape(out_array))
                for i in range(len(out_array[:,0])):
                    log_state=list(out_array[i,:])
                    wlog.writerow(log_state)
                    newfile.flush()
                print("\nFile saved as:", outfile)
                newfile.close()
            else: count+=1
            
        # except: print("Could not read file:",name_file)
    #print shape(out_array)
    #print out_array[1:5,:]
    if len(files)==1 or tag != "":
        for i in range(len(out_array[:,0])):
            log_state=list(out_array[i,:])
            wlog.writerow(log_state)
            newfile.flush()


        newfile.close()
        print("\nFile saved as:", outfile)



# import lib_utilities
# lib_utilities.write_ts_te_table("/Users/daniele/Desktop/try/tries",0)



def calc_marginal_likelihood(infile,burnin,extract_mcmc=1):
    sys.path.append(infile)
    direct="%s/*.log" % infile # PyRateContinuous
    files=glob.glob(direct)
    files=sort(files)
    if len(files)==0:
        print("log files not found.")
        quit()
    else: print("found", len(files), "log files...\n")    
    out_file="%s/marginal_likelihoods.txt" % (infile)
    newfile =open(out_file,'w') # python2 'wb'
    tbl_header = "file_name\tmodel\tTI_categories\treplicate"
    for f in files:
        try: 
        #if 2>1:
            t=loadtxt(f, skiprows=max(1,burnin))
            input_file = os.path.basename(f)
            name_file = os.path.splitext(input_file)[0]
            
            try:
                replicate = 0
                model = "NA"
                for i in name_file.split('_'):
                    if "BD" in i or "lin" in i or "exp" in i:
                        model = i
                    try: 
                        replicate = int(i)
                    except: pass
            except: replicate =0 
            #model = name_file.split('_')[-1]
            #if "exp.log" in f: model = "Exponential"
            #else: model = "Linear"
            
            head = np.array(next(open(f)).split()) # should be faster
            head_l = list(head)
            
            L_index = [head_l.index(i) for i in head_l if "lik" in i]
            
            for i in L_index: tbl_header+= ("\t"+head[i])
            
            if f==files[0]: newfile.writelines(tbl_header)
            
            
            #try:
            #    like_index = np.where(head=="BD_lik")[0][0]
            #except(IndexError): 
            #    like_index = np.where(head=="likelihood")[0][0] 
            try: 
                temp_index = np.where(head=="temperature")[0][0]
            except(IndexError): 
                temp_index = np.where(head=="beta")[0][0]
    
            z=t[:,temp_index] # list of all temps
            y=collections.Counter(z)
            x=list(y) # list of unique values
            x=sort(x)[::-1]
            l=zeros(len(x))
            s=zeros(len(x))
            l_str,s_str="",""
            for like_index in L_index:
                for i in range(len(x)): 
                    l[i]=mean(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
                    s[i]=std(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
                    #l_str += "\t%s" % round(l[i],3)
                    #s_str += "\t%s" % round(s[i],3)
                mL=0
                for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(x[i]-x[i+1]) # Beerli and Palczewski 2010
                #ml_str="\n%s\t%s\t%s\t%s%s%s" % (name_file,round(mL,3),replicate,model    ,l_str,s_str)
                n_categories = len(x)
                if like_index == min(L_index): ml_str="\n%s\t%s\t%s\t%s\t%s" % (name_file,model,n_categories,replicate,round(mL,3))
                else: ml_str += "\t%s" % (round(mL,3))
            newfile.writelines(ml_str)
            newfile.flush()
            if extract_mcmc==1:
                out_name="%s/%s_cold.log" % (infile,name_file)
                newfile2 =open(out_name,'w') # python2 'wb'
                
                t_red = t[(t[:,temp_index]==1).nonzero()[0]]
                newfile2.writelines( "\t".join(head_l)  )
                for l in range(shape(t_red)[0]):
                    line= ""
                    for i in t_red[l,:]: line+= "%s\t" % (i)
                    newfile2.writelines( "\n"+line  )
                #
                #l[i]=mean(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
                    #
                newfile2.close()
            
            
            
            
            sys.stdout.write(".")
            sys.stdout.flush()
        except:
            print("\n WARNING: cannot read file:", f, "\n\n")

    newfile.close()



def parse_hsp_logfile(logfile,burnin=100):
    t=np.loadtxt(logfile, skiprows=max(1,burnin))
    head = next(open(logfile)).split()
    
    baseline_L = mean(t[:,4])
    baseline_M = mean(t[:,5])
        
    l_indexL = [head.index(i) for i in head if "Gl" in i]
    l_indexM = [head.index(i) for i in head if "Gm" in i]
    
    C = list(head[4])
    fixed_focal_clade = ""
    for i in range(1,len(C)):
        fixed_focal_clade+=C[i]
    
    fixed_focal_clade = int(fixed_focal_clade) 
    k_indexL = [head.index(i) for i in head if "Wl" in i]
    k_indexM = [head.index(i) for i in head if "Wm" in i]
    
    gl,gm,kl,km = list(),list(),list(),list()
    for i in range(len(l_indexL)):
        gl.append(mean(t[:,l_indexL[i]]))
        kl.append(mean(t[:,k_indexL[i]]))
        gm.append(mean(t[:,l_indexM[i]]))
        km.append(mean(t[:,k_indexM[i]]))
    
    return(fixed_focal_clade,baseline_L,baseline_M,np.array(gl),np.array(gm),np.array(kl),np.array(km))
 


def parse_hsp_logfile_HPD(logfile,burnin=100):
    t=np.loadtxt(logfile, skiprows=max(1,burnin))
    head = next(open(logfile)).split()
    
    # get col indexes
    L0_index = 4
    M0_index = 5
    l_indexL = [head.index(i) for i in head if "Gl" in i]
    l_indexM = [head.index(i) for i in head if "Gm" in i]
    k_indexL = [head.index(i) for i in head if "Wl" in i]
    k_indexM = [head.index(i) for i in head if "Wm" in i]
    
    # get number of focal clade
    C = list(head[L0_index])
    fixed_focal_clade = ""
    for i in range(1,len(C)):  # so if head[L0_index]== "l12": C=["l","1","2"] and fixed_focal_clade = "12"
        fixed_focal_clade+=C[i]
    
    fixed_focal_clade = int(fixed_focal_clade) 
    
        
    print("\nGetting posterior parameter values...")
    # store l0,m0,gl,gm values for each MCMC iteration
    L0_list,M0_list,gl_list,gm_list = list(),list(),list(),list()

    for j in range(shape(t)[0]):
        # baseline rates
        L0 = t[j,L0_index]
        M0 = t[j,M0_index]
        # G parameters
        gl = t[j,l_indexL]
        gm = t[j,l_indexM]
        
        L0_list.append(L0)
        M0_list.append(M0)
        gl_list.append(gl)
        gm_list.append(gm)

    L0_list = np.array(L0_list)
    M0_list = np.array(M0_list)
    gl_list = np.array(gl_list)
    gm_list = np.array(gm_list)
    #print np.shape(gl_list)
    
    # get posterior estimate of k
    kl,km = list(),list()
    for i in range(len(l_indexL)):
        kl.append(mean(t[:,k_indexL[i]]))
        km.append(mean(t[:,k_indexM[i]]))
    
    gl,gm = list(),list()
    for i in range(len(l_indexL)):
        gl.append(mean(t[:,l_indexL[i]]))
        gm.append(mean(t[:,l_indexM[i]]))
    
    return(fixed_focal_clade,L0_list,M0_list,gl_list,gm_list,np.array(kl),np.array(km))


def get_mode(data):
    # determine bins Freedman-Diaconis rule
    iqr = np.subtract(*np.percentile(data,[75,25])) # interquantile range
    h_temp = 2 * iqr * len(data) **(-1./3) # width
    h = max(h_temp,0.0001)
    n_bins= int((max(data)-min(data))/h)
    hist=np.histogram(data,bins=n_bins)
    return hist[1][np.argmax(hist[0])] # modal value



#### CHECK TAXA NAMES FOR TYPOS
def calc_diff_string(a,b):
    s = a==b
    score = np.float(len(s[s == True]))/len(s)
    s_diff = len(s[s == False])
    return score, s_diff
    
def get_score(a,b,max_length_diff):
    if a==b is True: score = 1
    else:
        a=a.lower() 
        b=b.lower() 
        if a==b is True: score = 1 # no matter upper/lower case
        else:
            a1 = np.array(list(a))
            b1 = np.array(list(b))
            if len(a1)==len(b1): # if same length assume no missing/extra letters
                score, s_diff = calc_diff_string(a1,b1)
            elif np.abs(len(a1)-len(b1)) > max_length_diff:
                score, s_diff = 0, len(b1)
            else:
                l_a =a1[np.array(list(itertools.combinations(np.arange(len(a1)),min(len(a1),len(b1)))))]
                l_b =b1[np.array(list(itertools.combinations(np.arange(len(b1)),min(len(a1),len(b1)))))]
                s = l_a==l_b
                s_bin = s.astype(None) # convert True/False array into 1/0 array
                score = np.max(np.sum(s_bin,axis=1))/np.mean([len(a1),len(b1)])
                s_diff = np.abs(len(a1)-len(b1)) + ( min(len(a1),len(b1)) - np.max(np.sum(s_bin,axis=1)) )
    if score==1: s_diff=0
    return score, s_diff

def reduce_log_file(log_file,burnin=1): # written by Tobias Hofmann (tobias.hofmann@bioenv.gu.se)
    print(log_file)
    target_columns = ["it","posterior","prior","PP_lik","BD_lik","q_rate","alpha","k_birth","k_death","root_age","death_age","q_","tot_length"]
    workdir = os.path.dirname(log_file)
    if workdir=="": workdir= self_path
    input_file = os.path.basename(log_file)
    name_file = os.path.splitext(input_file)[0]
    outfile = "%s/%s_reducedLog.log" %(workdir,name_file)
    output = open(outfile, "w")
    outlog=csv.writer(output, delimiter='\t')
    print("Parsing header...")
    head = next(open(log_file)).split()
    w=[head.index(x) for x in head if x in target_columns]
    
    w = []
    col_names = []
    for i in range(len(head)):
        if head[i] in target_columns or "q_" in head[i]:
            w.append(i)
            col_names.append(head[i])
    print(w)
    print("Reading mcmc log file...")
    #log_file = "%s" % (log_file)
    print(log_file)
    tbl = np.loadtxt(log_file,skiprows=burnin,usecols=(w))
    print(np.shape(tbl))    
    outlog.writerow(col_names)
    for i in tbl: outlog.writerow(list(i))
    print("The reduced log file was saved as: %s\n" % (outfile))
    

def write_des_in(out_list, reps, all_taxa_list, taxon, time, input_wd, filename):
    for i in range(reps):
        in_file = "%s/%s_%s.txt" % (input_wd, filename, i + 1)
        w_in_file = open(in_file, "w")
        writer = csv.writer(w_in_file, delimiter='\t')
        head = [taxon] + list(time)
        _ = writer.writerow(head)
        for a in range(out_list[i].shape[0]):
            row = [all_taxa_list[i][a]] + list(out_list[i][a,:])
            _ = writer.writerow(row)
        w_in_file.flush()
        os.fsync(w_in_file)

def des_in(x, recent, input_wd, filename, taxon = "scientificName", area = "higherGeography", age1 = "earliestAge", age2 = "latestAge", binsize = 5., reps = 3, trim_age = [], data_in_area = []):
    rece = np.genfromtxt(recent, dtype = str, delimiter='\t')
    rece_names = rece[0,:]
    rece = np.unique(rece[1:,:], axis = 0)
    rece_names_area = rece_names == area
    rece_taxa = rece[:,rece_names == taxon].flatten()
    areas = np.unique(rece[:,rece_names_area])
    area_recent = np.zeros(rece.shape[0], dtype=int)
    if data_in_area == 0:
        area_recent[np.array(rece[:, rece_names_area] == areas[0]).flatten()] = 1
        area_recent[np.array(rece[:, rece_names_area] == areas[1]).flatten()] = 2
    else:
        area_recent = area_recent + 3
    out_list = []
    all_taxa_list = []
    dat = np.genfromtxt(x, dtype=str, delimiter='\t')
    dat_names = dat[0,:]
    dat = dat[1:,:]
    dat_names_taxon = np.where(dat_names == taxon)
    dat_taxa = dat[:,dat_names_taxon].flatten()
    dat_names_area = np.where(dat_names == area)
    dat_names_age1 = np.where(dat_names == age1)
    dat_names_age2 = np.where(dat_names == age2)
    dat_ages = dat[:,np.concatenate((dat_names_age1, dat_names_age2), axis = None)]
    dat_ages = dat_ages.astype(float)
    max_age = np.max(dat_ages)
    if np.array(trim_age) > 0:
        max_age = np.array(trim_age)
    cutter = np.arange(0., max_age + binsize, binsize)
    cutter_len = len(cutter)
    for i in range(reps):
        age_ran = np.zeros(dat.shape[0])
        for y in range(dat.shape[0]):
            age_ran[y] = np.random.uniform(dat_ages[y, 0], dat_ages[y, 1], 1)
        binnedage = np.digitize(age_ran, cutter) # Starts with 1!
        area_fossil = np.zeros(dat.shape[0], dtype=int)
        area_fossil[np.array(dat[:,dat_names_area] == areas[0]).flatten()] = 1
        if data_in_area == 0:
            area_fossil[np.array(dat[:,dat_names_area] == areas[1]).flatten()] = 2
        elif data_in_area == 1:
            area_fossil = area_fossil + 1
        all_taxa = np.concatenate((np.unique(dat_taxa), np.unique(rece_taxa)), axis = None)
        all_taxa = np.unique(all_taxa)
        # First column is the most recent time bin and we reverse this latter
        out = np.zeros((len(all_taxa), cutter_len + 1))
        if data_in_area == 1:
            out = out + 2
        elif data_in_area == 2:
            out = out + 1
        for a in range(len(all_taxa)):
            idx = np.where(dat_taxa == all_taxa[a])
            idx = np.array(idx).flatten()
            if idx.size > 0:
                binnedage_taxon = binnedage[idx]
                area_taxon = area_fossil[idx]
                younger_max_age = binnedage_taxon < cutter_len
                binnedage_taxon = binnedage_taxon[younger_max_age]
                area_taxon = area_taxon[younger_max_age]
                for b in binnedage_taxon:
                    area_taxon_b = np.unique(area_taxon[binnedage_taxon == b])
                    if len(area_taxon_b) == 1 and data_in_area == 0:
                        area_code = area_taxon_b
                    else:
                        area_code = 3
                    out[a,b] = area_code
                    if data_in_area == 0:
                        if b == np.max(binnedage_taxon) and b != cutter_len:
                            out[a, (b + 1):(cutter_len + 1)] = np.nan
                    else:
                        out[a, cutter_len] = np.nan
            if np.isin(all_taxa[a], rece_taxa):
                idx = np.where(rece_taxa == all_taxa[a])
                area_taxon = area_recent[idx]
                if len(area_taxon) == 1:
                    area_code = area_taxon
                else:
                    area_code = 3
                out[a,0] = area_code
        out = out[:,::-1]
        # Remove taxa without fossils
        any_record = np.nansum(out[:,:-1], axis = 1) != 0
        out = out[any_record,:]
        all_taxa = all_taxa[any_record]
        out_list.append(out)
        all_taxa_list.append(all_taxa)
    # Truncate columns of out_list without records
    colsum_out = np.zeros((reps, cutter_len + 1))
    for i in range(reps):
        colsum_out[i,:] = np.nansum(out_list[i], axis = 0)
    no_records_yet = np.cumsum(np.sum(colsum_out, axis = 0)) == 0
    keep_rows = np.sum(no_records_yet) - 1
    if keep_rows > 1:
        for i in range(reps):
            out_list[i] = out_list[i][:,keep_rows:]
        cutter = cutter[:-keep_rows] # Truncate cutter to dim2 of out_list
    tdiff = np.diff(cutter)[0]
    time = np.arange(0., np.max(cutter) + tdiff + tdiff/1000., tdiff)
    time[1:] = time[1:] - tdiff/2
    time = time[::-1]
    write_des_in(out_list, reps, all_taxa_list, taxon, time, input_wd, filename)
    return out_list, time
