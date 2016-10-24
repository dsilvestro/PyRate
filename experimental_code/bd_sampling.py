from numpy import *
import numpy as np
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

nbins  = 100000
xLim   = 100
x_bins = np.linspace(0.01,xLim,nbins) 
x_bin_size = x_bins[1]-x_bins[0]

def bd_lik_per_species(s,e,l,m):
	d=s-e
	extant= np.ones(len(e))
	extant[e==0]=0
	return log(l)+log(m)*extant-(l+m)*d

def birth_lik_per_species(s,e,l,m):
	d=s-e
	return sum(log(l)-(l)*d)

def exp_pdf(d,m):
	rate=m
	return log(rate)-rate*d
		
#def exp_pdf(d,m):
#	rate=1./m
#	return rate*exp(-rate*d)


def sampling_prob(d,q):
	return log(1-exp(-q*d))

def simulate_bd_lineages(L,M,timesL, timesM,root,scale=100.,s_species=1, maxSP=1000):
	ts=list()
	te=list()
	l_t=L[0]
	m_t=M[0]
	if timesL==0: timesL=np.array([-root,0])
	if timesM==0: timesM=np.array([-root,0])
	
	L,M,root=L/scale,M/scale,int(root*scale)
	
	for i in range(s_species): 
		ts.append(root)
		te.append(0)
		
	for t in range(root,0): # time
		for j in range(len(timesL)-1):
			if -t/scale<=timesL[j] and -t/scale>timesL[j+1]: l=L[j]
		for j in range(len(timesM)-1):
			if -t/scale<=timesM[j] and -t/scale>timesM[j+1]: m=M[j]
		
		TE=len(te)
		for j in range(TE): # extant lineages
			if te[j]==0:
				ran=random.random()
				if ran<l: 
					te.append(0) # add species
					ts.append(t) # sp time
				elif ran>l and ran < (l+m): # extinction
					te[j]=t
	te=array(te)
	return -np.array(ts)/scale, -np.array(te)/scale

def death_lik(s,e,l,m):
	d=s-e
	extant= np.ones(len(e))
	extant[e==0]=0
	return log(m)*extant-(m)*d


def bd_lik_sampling(s,e,l,m,q):	# corrected BD + Sampling	
	P = exp(bd_lik_per_species(x_bins,np.zeros(len(x_bins)),l,m) + sampling_prob(x_bins,q))
	extinct_lik = sum(bd_lik_per_species(s[e>0],e[e>0],l,m) + sampling_prob(d[e>0],q) - log(sum(P)*x_bin_size)) 
	extant_lik  = sum(np.array([log(sum(P[x_bins>j]*x_bin_size)) for j in s[e==0] ]) - log(sum(P)*x_bin_size))
	corrected_BD_lik = sum(extinct_lik+extant_lik)
	return corrected_BD_lik

def death_lik_sampling(s,e,l,m,q):	# corrected BD + Sampling	
	P = exp(death_lik(x_bins,np.zeros(len(x_bins)),l,m) + sampling_prob(x_bins,q))
	extinct_lik = sum(death_lik(s[e>0],e[e>0],l,m) + sampling_prob(d[e>0],q) - log(sum(P)*x_bin_size)) 
	extant_lik  = sum(np.array([log(sum(P[x_bins>j]*x_bin_size)) for j in s[e==0] ]) - log(sum(P)*x_bin_size))
	corrected_BD_lik = sum(extinct_lik+extant_lik)
	return corrected_BD_lik




l=np.array([0.4])
m=np.array([0.2])
while True:
	s,e = simulate_bd_lineages(l,m,timesL=0,timesM=0,root=-30)
	if len(s)>500 and len(s)<2000: break
#print s, e

#s=np.random.exponential(1./0.3,1000)+0.1
#e=np.zeros(1000)+0.1


print len(s), len(e[e>0]), max(s-e)
print "MLE speciation:", len(s)/sum(s-e)
print "MLE extinction:", len(e[e>0])/sum(s-e)

q=.25

d=s-e
k = np.random.poisson(q*d, len(d))
print "sampled taxa:", len(k[k>0])

# subset data
s=s[k>0]
e=e[k>0]
print "MLE speciation:", len(s)/sum(s-e)
print "MLE extinction:", len(e[e>0])/sum(s-e)

#q=sum(k)/sum(s-e)

for i in np.linspace(0.1,1,10):
	d=s-e            
	# pdf Exp distribution   # 1- cdf Exp
	uncorrected_lik = round(sum(exp_pdf(d[e>0],i))+ sum(log( exp(-i*d[e==0])  )),2)

	# pdf Exp distribution   # 1- cdf Exp
	uncorrected_BD_lik = round(sum(bd_lik_per_species(s,e,l,i)),2)

	# corrected Exp + Sampling
	P = exp(exp_pdf(x_bins,i) + sampling_prob(x_bins,q))
	extinct_lik = sum(exp_pdf(d[e>0],i)+sampling_prob(d[e>0],q) - log(sum(P)*x_bin_size)) 
	extant_lik  = sum(np.array([log(sum(P[x_bins>j]*x_bin_size)) for j in s[e==0] ]) - log(sum(P)*x_bin_size))
	corrected_lik = sum(extinct_lik+extant_lik)

	# corrected BD + Sampling	
	P = exp(bd_lik_per_species(x_bins,np.zeros(len(x_bins)),l,i) + sampling_prob(x_bins,q))
	extinct_lik = sum(bd_lik_per_species(s[e>0],e[e>0],l,i) + sampling_prob(d[e>0],q) - log(sum(P)*x_bin_size)) 
	extant_lik  = sum(np.array([log(sum(P[x_bins>j]*x_bin_size)) for j in s[e==0] ]) - log(sum(P)*x_bin_size))
	corrected_BD_lik = sum(extinct_lik+extant_lik)


	print round(i,2),uncorrected_lik,round(corrected_lik,2), uncorrected_BD_lik,round(corrected_BD_lik,2)


quit()
#


lA,mA=l[0],m[0]
likA=bd_lik_sampling(s,e,lA,mA,q)+birth_lik_per_species(s,e,lA,mA)
for i in range(10000):
	l=abs(lA+np.random.normal(0,0.1))
	m=abs(mA+np.random.normal(0,0.1))
	#lik = bd_lik_per_species(s,e,l,m)+sampling_prob(s-e,q)
	#lik= exp_pdf(s-e,m)+sampling_prob(s-e,q) # ADD INTEGRAL
	lik = death_lik_sampling(s,e,l,m,q)+birth_lik_per_species(s,e,l,m)
	if sum(lik)-sum(likA) >= log(np.random.random()):
		lA,mA=l,m
		likA=lik
	if i%10 ==0: print i, sum(likA),lA,mA