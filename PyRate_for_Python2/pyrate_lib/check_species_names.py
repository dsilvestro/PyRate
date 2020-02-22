import itertools
from numpy import *
import numpy as np
import scipy, scipy.stats
from scipy.special import gamma
import os, sys, csv, time
import string
import unicodedata
encoding = "utf-8"

rseed = 1111
np.random.seed(rseed)
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

def calc_diff_string(a,b):
	s = a==b
	score = np.float(len(s[s == True]))/len(s)
	s_diff = len(s[s == False])
	if s_diff==2: # teste inverted letters
		s1 = np.sort(a)==np.sort(b)
		if len(s1[s1 == False]) == 0:
			score,s_diff= 0.999,1.
	return score, s_diff

def fix_replace_str(a):
	s = a+"#"
	s = string.replace(s, "yi", "i")	
	s = string.replace(s, "ae", "e")
	s = string.replace(s, "oe", "e")
	s = string.replace(s, "ou", "u")
	#s = string.replace(s, "ll", "l")
	#s = string.replace(s, "mm", "m")
	s = string.replace(s, "k", "c")
	s = string.replace(s, "np", "mp")
	s = string.replace(s, "pseudo", "pseud")
	s = string.replace(s, "iens", "ens")
	s = string.replace(s, "schu", "shu")
	s = string.replace(s, "sche", "she")
	
	
	
	
	
	# fix endings
	s = string.replace(s, "ys#"  , "is#")
	s = string.replace(s, "ius#" , "is#")
	s = string.replace(s, "es#"  , "is#")
	s = string.replace(s, "e#"  ,  "is#")
	s = string.replace(s, "ii#"  , "is#")
	s = string.replace(s, "i#"   , "is#")
	# s = string.replace(s, "inis#", "inis#")
	# s = string.replace(s, "inus" , "inis#")
	s = string.replace(s, "as#"  , "is#")
	s = string.replace(s, "us#"  , "is#")
	s = string.replace(s, "a#"   , "is#")
	s = string.replace(s, "um#"  , "is#")
	s = string.replace(s, "ei#"  , "is#")
	# finally
	s = string.replace(s, "#"  , "")
	return s

def remove_accents(input_str):
	nfkd_form = unicodedata.normalize('NFKD', input_str)
	only_ascii = nfkd_form.encode('ASCII', 'ignore')
	return only_ascii

def get_score_trained(a,b,max_length_diff=2):
	score = -1 
	if a==b: # identical strings
		score = 1 # no matter upper/lower case
		s_diff = 0
	else:
		# remove case issues
		a=a.lower() 
		b=b.lower()
		# remove accents
		unicode_string_a = a.decode(encoding)
		unicode_string_b = b.decode(encoding)
		a=remove_accents(unicode_string_a)
		b=remove_accents(unicode_string_b)
		if a==b:
			score = 0.99
			s_diff =0			
		# min/max/maj
		if "min" in a:
			if "max" in b or "maj" in b:
				score =0.1
				s_diff =2
		if "min" in b:
			if "max" in a or "maj" in a:
				score = 0.1
				s_diff =2
		if a.startswith("uni"):
			if b.startswith("di") or b.startswith("bi"):
				score =0.1
				s_diff =2
		if b.startswith("di") or b.startswith("bi"):
			if a.startswith("uni"):
				score = 0.1
				s_diff =2
		if score < 0:
			# try substitutions
			a = fix_replace_str(a)
			b = fix_replace_str(b)
			if a==b:
				score =0.99
				s_diff =0
			else:
				a1 = np.array(list(a))
				b1 = np.array(list(b))
				if len(a1)==len(b1): # if same length assume no missing/extra letters
					score, s_diff = calc_diff_string(a1,b1)
				elif np.abs(len(a1)-len(b1)) > max_length_diff:
					score, s_diff = 0, len(b1)
				#else:
				#	l_a =a1[np.array(list(itertools.combinations(np.arange(len(a1)),min(len(a1),len(b1)))))]
				#	l_b =b1[np.array(list(itertools.combinations(np.arange(len(b1)),min(len(a1),len(b1)))))]
				#	s = l_a==l_b
				#	s_bin = s.astype(None) # convert True/False array into 1/0 array
				#	score = np.max(np.sum(s_bin,axis=1))/np.mean([len(a1),len(b1)])
				#	s_diff = np.abs(len(a1)-len(b1)) + ( min(len(a1),len(b1)) - np.max(np.sum(s_bin,axis=1)) )

				else: # 'align' letters to fix gaps
					max_length = max(len(a1),len(b1))
					if abs(len(a1)-len(b1))==2: # if the difference is the first two letters it's different taxa
						if len(a1) == max_length:
							s = a1[2:] ==b1
							if sum(s.astype(None))==len(b1): 
								return 0.8, 2
						if len(b1) == max_length:
							s = b1[2:] ==a1
							if sum(s.astype(None))==len(a1):
								return 0.8, 2

					if abs(len(a1)-len(b1))==1: # if the difference is the first letter it's possibly different taxa
						if len(a1) == max_length:
							s = a1[1:] ==b1
							if sum(s.astype(None))==len(b1): 
								return 0.9, 1
						if len(b1) == max_length:
							s = b1[1:] ==a1
							if sum(s.astype(None))==len(a1):
								return 0.9, 1
							
					#if len(a1) < max_length:
					#	a1 = np.append(a1, np.array(["_" for i in range(max_length-len(a1))]))
					#if len(b1) < max_length:
					#	b1 = np.append(b1, np.array(["_" for i in range(max_length-len(b1))]))
					
					#qq=np.array(list(itertools.permutations(np.arange(len(a1)))))
					#l_a =a1[qq]
					#l_b =b1[qq]
				
					l_a =a1[np.array(list(itertools.combinations(np.arange(len(a1)),min(len(a1),len(b1)))))]
					l_b =b1[np.array(list(itertools.combinations(np.arange(len(b1)),min(len(a1),len(b1)))))]
					s = l_a==l_b
					s_bin = s.astype(None) # convert True/False array into 1/0 array
					score = np.max(np.sum(s_bin,axis=1))/np.mean([len(a1),len(b1)])
					s_diff = np.abs(len(a1)-len(b1)) + ( min(len(a1),len(b1)) - np.max(np.sum(s_bin,axis=1)) )

	return score, s_diff


def check_taxa_names(w,out_file_name="output.txt"):
	words = np.unique(w)
	# print "\nTaxa names with possible misspells (if any) will be listed below..."
	
	logfile = open(out_file_name , "wb") 
	wlog=csv.writer(logfile, delimiter='\t')
	head="taxon1\ttaxon2\trank" 
	wlog.writerow(head.split('\t'))
	logfile.flush()
	
	word_combinations = itertools.combinations(words,2)
	comb = len(list(itertools.combinations(words,2)))
	print "Testing", comb, "combinations... (progress printed below)"
	start_time = time.time()
	# sensitivity settings
	max_length_diff = 2 # maximum allowed difference between string lengths
	threshold_score = 0.7
	threshold_s_diff = 3
	all_scores = []
	j=0.
	for w in word_combinations: 
		try:
			taxon1 = w[0]
			taxon2 = w[1]	
			#score_all, diff_all = get_score_trained(taxon1,taxon2,max_length_diff)
			# GENUS
			a = taxon1.split("_")[0]
			b = taxon2.split("_")[0]
			score_genus, diff_genus = get_score_trained(a,b,max_length_diff)
			#print a,b,score_genus
			# SPECIES
			if len(taxon1.split("_"))>1 and len(taxon2.split("_"))>1:
				a = taxon1.split("_")[1]
				b = taxon2.split("_")[1]
				score_species, diff_species = get_score_trained(a,b,max_length_diff)
			else: score_species, diff_species = score_genus,0	
			s_diff = diff_genus+diff_species
			score_all, diff_all =  (score_genus+score_species)/2., s_diff
			if (score_genus+score_species)<2:
				if score_all > threshold_score and diff_all <= threshold_s_diff:
					if np.mean([score_genus,score_species]) > threshold_score and s_diff <= threshold_s_diff:
						all_scores.append([taxon1, taxon2,round(score_all,3),round(score_genus,3),
						round(score_species,3),int(s_diff)])
			j+=1
			if j % 100000==0: 
				progress_percentage = round(100*(j/comb),2)
				total_time = ((time.time()-start_time)*100/progress_percentage)/60.
				time_left = total_time-(time.time()-start_time)/60
				print progress_percentage,"%", int(time_left),"min left"
		except: print taxon1, taxon2
	all_scores = np.array(all_scores)
	# top hits:
	#print all_scores
	score_float = all_scores[:,2].astype(float)
	diff_int    = all_scores[:,5].astype(int)
	if len(all_scores)==0: sys.exit("No typos found!")
	th1,th2,rank = 0.99,0,0
	passed = np.array([])
	
	"""
	Write to file with threshold values 1, 2, 3
	
	"""
	while True:
		pass1 = (score_float>=th1).nonzero()[0]
		pass2 = (diff_int<=th2).nonzero()[0]
		res = np.union1d(pass1,pass2)
		for i in res: 
			if i not in passed: 
				log_line = list(all_scores[i][0:2])+[rank]
				wlog.writerow(log_line)
				logfile.flush()
		if len(res)==0: print "No typos found!"
		passed = np.union1d(res,passed)
		if len(passed)==len(all_scores): break
		th1 -= 0.05
		rank += 1
		if rank % 2. ==0: th2 += 1


def run_name_check(fossil_occs_file):
	print "\nParsing input data..."
	occs_names = np.loadtxt(fossil_occs_file,delimiter="\t",usecols=(0),skiprows=1,dtype="string")
	unique_names = np.unique(occs_names)
	print "Found %s unique taxa names" % (len(unique_names))
	# run typos-check on this list
	w = [s.replace(" ", "_") for s in occs_names]
	out_file_name = os.path.splitext(fossil_occs_file)[0]+"_scores.txt"
	check_taxa_names(w,out_file_name)
	print "The results (if any potential typos were found) are written here:"
	print out_file_name
	print """"Note that names of ranks 0 and 1 are the most likely cases of misspellings, \
whereas ranks 2 and 3 are most likely truly different names. \
This algorithm does NOT check for synonyms!\n"""

