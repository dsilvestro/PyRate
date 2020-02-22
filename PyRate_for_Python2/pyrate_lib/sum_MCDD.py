#!/usr/bin/env python 
import sys
import numpy as np
import os.path
import glob
from numpy import *
import argparse

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-path', type=str,   help='path to log files', default=0, metavar=0)
p.add_argument('-tag', type=str,   help='path to log files', default="", metavar="")
p.add_argument('-c', type=int,   help='no. clades to be combined', default=1, metavar=1)


args = p.parse_args()



def calcHPD(data, level) :
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
	return (d[i], d[i+nIn-1])



infile=args.path
path_dir = infile
sys.path.append(infile)
#direct="%s/*gibbs.log" % infile

clades = args.c


model_tag=args.tag

out_list =[]
for clade in range(clades): out_list.append("c"+str(clade)+model_tag)
	
f_nameL = path_dir +"/Gl_summary%s.txt" % (model_tag)
f_nameM = path_dir +"/Gm_summary%s.txt" % (model_tag)
outfileL = open(f_nameL,'wb')
outfileM = open(f_nameM,'wb')
outfileL.writelines("from	to	mean	median	m	M	support")
outfileM.writelines("from	to	mean	median	m	M	support")

clade=0

print out_list

for out in out_list:
	direct="%s/*%s*.log" % (infile, out)
	files=glob.glob(direct)
	files=sort(files)
	f_num=0
	for f in files:
		file_name =  os.path.splitext(os.path.basename(f))[0]
		print file_name
	
		f_name = path_dir +"/"+ file_name.split('_')[1] + "_" + file_name.split('_')[-2] + ".txt"
                if 1==1:
			t=loadtxt(f, skiprows=max(1,100))
			head = next(open(f)).split()
			l_indexL = [head.index(i) for i in head if "Gl%s_" % (clade) in i]
			l_indexM = [head.index(i) for i in head if "Gm%s_" % (clade) in i]
			
			k_indexL = [head.index(i) for i in head if "kl%s_" % (clade) in i]
			k_indexM = [head.index(i) for i in head if "km%s_" % (clade) in i]
			
			
			j=1
			for i in l_indexL:
				par = t[:,i]
				k_par = t[:,k_indexL[j-1]]
				hpd = np.around(calcHPD(par, .95), decimals=6)
				#mode = round(get_mode(par),3)
				mean_par = round(mean(par),6)
				median_par = round(median(par),6)
				mean_k= round(median(k_par),6)
				if f_num==0: 
					outfileL.writelines("\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % (j,clade+1, mean_par,median_par,hpd[0],hpd[1],mean_k))
				else: 
					outfileL.writelines("\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % (j,clade+1, mean_par,median_par,hpd[0],hpd[1],mean_k))
				j+=1
			j=1
			for i in l_indexM:
				par = t[:,i]
				k_par = t[:,k_indexM[j-1]]
				hpd = np.around(calcHPD(par, .95), decimals=6)
				mean_par = round(mean(par),6)
				median_par = round(median(par),6)
				mean_k= round(median(k_par),6)
				if f_num==0: 
					outfileM.writelines("\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % (j,clade+1, mean_par,median_par,hpd[0],hpd[1],mean_k))
				else: 
					outfileM.writelines("\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % (j,clade+1, mean_par,median_par,hpd[0],hpd[1],mean_k))
				j+=1
			f_num+=1
                else: pass
	clade+=1
outfileL.close()
outfileM.close()

