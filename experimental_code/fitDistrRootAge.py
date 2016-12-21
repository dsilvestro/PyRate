def fitDistrRootAge(d): # add burnin, column tag, etc. 
	d = np.loadtxt(file) #scipy.stats.halflogistic(1,1).rvs(500)
	print "\n\n",file
	data=d #(d-min(d))+0.00001
	data_mod=data-min(data)+.00000000001
	
	print "\nLAPLACE"
	LAP= scipy.stats.laplace.fit(data)
	print "loc:", LAP[0], "scale:", LAP[1] #, "cutoff:",min(d)
	lik1=sum(scipy.stats.laplace.logpdf(data,LAP[0],LAP[1]))
	print "lik:",lik1
	
	print "\nGAMMA"
	LAP= scipy.stats.gamma.fit(data_mod,floc=0) # force loc=0
	print "shape:", LAP[0], "scale:", LAP[2], "offset:",min(d) 
	lik2=sum(scipy.stats.gamma.logpdf(data_mod,LAP[0],LAP[1],LAP[2]))
	print "lik:",lik2
	
	print "\nNORMAL"
	LAP= scipy.stats.norm.fit(data)
	print "loc:", LAP[0], "scale:", LAP[1] #, "cutoff:",min(d)
	lik3=sum(scipy.stats.norm.logpdf(data,LAP[0],LAP[1]))
	print "lik:",lik3
	
	print "\nLOGNORMAL"
	LAP= scipy.stats.lognorm.fit(data_mod ,floc=0)
	#print "shape:", LAP[0], "loc:", LAP[1] , "scale:",LAP[2]
	print "mean:", log(LAP[2]), "sd:", LAP[0], "offset:", min(d)
	lik4=sum(scipy.stats.lognorm.logpdf(data_mod,LAP[0],LAP[1],LAP[2]))
	print "lik:",lik4
	
	L=np.array([lik1,lik2,lik3,lik4])
	print L-max(L)
