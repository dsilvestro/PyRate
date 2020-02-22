#include "FastPyRateC.h"

#include <assert.h>
#include <utility>
#include <algorithm>
#include <iostream>
#include <map>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#define LOG_PLUS(x,y) (x>y ? x+log(1.+exp(y-x)) : y+log(1.+exp(x-y)))


/******************************************************************************************/
/******************************      Utils functions     **********************************/
/******************************************************************************************/

// Count number of event in a given time bin
unsigned int countEvent(const int nSpecies, const double start, const double end, const std::vector<double> &t) {
	unsigned int cnt=0;
	for(size_t iS=0; iS<(size_t)nSpecies; ++iS) {
		//std::cout << iS << " :: " << ts[iS] << std::endl;
		if(t[iS] <= start && t[iS] > end){
			++cnt;
		}
	}
	return cnt;
}

// Count existence time in a given time bin
double getTimeInsideTF(double ts, double te, const double start, const double end) {
	// Check if ts is at least bigger than the end of the timeframe
	if(ts > end){
		// If so check if it is bigger than the start of the TF
		if(ts >= start) {
			// Then we have to change it to start
			ts = start;
		}
	} else {
		// we are not in TF
		return 0.;
	}

	// Check if te is at least smaller than the start of the TF
	if(te < start){
		// If so check if it is small than the end of the TF
		if(te <= end) {
			// Then we have to change it to end
			te = end;
		}
	} else {
		// we are not in TF
		return 0.;
	}

	return ts-te;
}

// Fast gamma (one)
double PyRateC_getLogGammaPDF(double value, double shape, double scale) {
	boost::math::gamma_distribution<> dist(shape, scale);
	return log(boost::math::pdf(dist, value));
}

// Fast gamma (multiple)
std::vector<double> PyRateC_getLogGammaPDF(std::vector<double> values, double shape, double scale) {
	boost::math::gamma_distribution<> dist(shape, scale);
	std::vector<double> results(values.size());
	for(size_t i=0; i<values.size(); ++i) {
		results[i] = log(boost::math::pdf(dist, values[i]));
	}
	return results;
}

// Compute mean Gamma Rate (Yang)
std::vector<double> computeGammaRate(size_t N_GAMMA, double alpha) {
	std::vector<double> gammaRates(N_GAMMA);

	double beta = alpha;
	boost::math::chi_squared_distribution<double> chi2Dist(2.*alpha);

	// Compute cutting point (Eq. 9) (could use gamma QUANTILE instead)
	std::vector<double> cPoints(1, 0.);
	for(size_t iG=1; iG<N_GAMMA; ++iG) {
		double cPoint = boost::math::quantile(chi2Dist, (double)iG/(double)N_GAMMA) / (2.*beta);
		cPoints.push_back(cPoint);
	}

	// Compute all lower incomplete gamma functions required in Eq. 10
	double alphaP1 = alpha + 1;
	std::vector<double> lowerIncGamma(1, 0.);
	for(size_t iL=1; iL<cPoints.size(); ++iL) {
		double liGamma = boost::math::gamma_p(alphaP1, cPoints[iL]*beta);
		lowerIncGamma.push_back(liGamma);
	}
	lowerIncGamma.push_back(1.);

	// Equation 10.
	for(size_t iG=0; iG<N_GAMMA; ++iG) {
		// alpha = beta, thus alpha/beta = 1.0 || k=iG
		gammaRates[iG] = ((double)N_GAMMA)*(lowerIncGamma[iG+1]-lowerIncGamma[iG]);
	}
	return gammaRates;
}

/******************************************************************************************/
/******************************       Init functions     **********************************/
/******************************************************************************************/

// Pre compute log factorials
std::vector<double> logFactorials;
std::vector <double> logFactorialFossilCntPerSpecie;
void determineLogFactorialFossilCntPerSpecie(std::vector< std::vector <double> > &aFossils) {

	// Get max nFossils
	size_t N_MAX_FOSSILS = 0;
	for(size_t iF=0; iF<aFossils.size(); iF++) {
		if(N_MAX_FOSSILS < aFossils[iF].size()) N_MAX_FOSSILS = aFossils[iF].size();
	}

	// Compute log factorial from 1 to N_MAX_FOSSILS+1
	N_MAX_FOSSILS += 1;
	logFactorials.assign(N_MAX_FOSSILS+1, 0);
	logFactorials[1] = log(1.0);
	for(size_t iL=2; iL<=N_MAX_FOSSILS; iL++) {
		double logFactorial = logFactorials[iL-1] + log((double)iL);
		logFactorials[iL] = logFactorial;
	}

	// Attribute log factorial to each specie as a function of the fossil occurence
	for(size_t iF=0; iF < aFossils.size(); ++iF) {
		size_t idLogFacto = aFossils[iF].size();
		if(aFossils[iF].back() == 0.) idLogFacto--;
		double logFactorial = logFactorials[idLogFacto];
		logFactorialFossilCntPerSpecie.push_back(logFactorial);
	}
}

// Pre compute fossil count per epochs
std::vector< std::vector <size_t> > fossilCountPerEpoch;
void determineFossilsOccurencePerEpoch(std::vector< double > &aEpochs, std::vector< std::vector <double> > &aFossils) {
	assert(!aFossils.empty());
	assert(!aEpochs.empty());

	for(size_t iS=0; iS<aFossils.size(); ++iS) { // For each specie
		std::vector<size_t>	fossilCount(aEpochs.size());
		for(size_t iE=0; iE<aEpochs.size()-1; ++iE) {
			double start = aEpochs[iE];
			double end = aEpochs[iE+1];

			fossilCount[iE] = countEvent(aFossils[iS].size(), start, end, aFossils[iS]);
		}
		fossilCountPerEpoch.push_back(fossilCount);
	}
}

// Init fossils
std::vector< std::vector <double> > fossils;
void PyRateC_setFossils(std::vector< std::vector <double> > aFossils) {
	fossils = aFossils;
	for(size_t iF=0; iF<fossils.size(); ++iF) {
		std::sort(fossils[iF].rbegin(), fossils[iF].rend());
	}
	determineLogFactorialFossilCntPerSpecie(fossils);
}

// Init epochs
void PyRateC_initEpochs(std::vector< double > aEpochs) {
	determineFossilsOccurencePerEpoch(aEpochs, fossils);
}

/******************************************************************************************/
/************************       BD_partial_lik functions     ******************************/
/******************************************************************************************/

void BD_partial_lik_by_rate(
  const std::vector<double> &ts,
  const std::vector<double> &te,
  const std::vector<double> &vTx,
  const std::vector<double> &rates,
  const std::string &par,
  std::vector<double> &results){

	int nSpecies = ts.size();

	for(size_t iR=0; iR<rates.size(); ++iR){
		double start = vTx[iR];
		double end = vTx[iR+1];

		unsigned int nEvent;
		if(par == "l"){
			nEvent = countEvent(nSpecies, start, end, ts);
		} else {
			nEvent = countEvent(nSpecies, start, end, te);
		}

		double totalBL = 0; // Total branch length
		for(size_t iS=0; iS<(size_t)nSpecies; ++iS) {
			totalBL += getTimeInsideTF(ts[iS], te[iS], start, end);
		}

		results.push_back(nEvent*log(rates[iR]) + -rates[iR]*totalBL);
	}

}

std::vector<double> PyRateC_BD_partial_lik(
  std::vector<double> ts,
  std::vector<double> te,
  std::vector<double> timeFrameL,
  std::vector<double> timeFrameM,
  std::vector<double> ratesL,
  std::vector<double> ratesM) {

	std::vector<double> results;

	BD_partial_lik_by_rate(ts, te, timeFrameL, ratesL, "l", results);
	BD_partial_lik_by_rate(ts, te, timeFrameM, ratesM, "m", results);

	return results;
}


/******************************************************************************************/
/**************************       HOMPP_lik functions     *********************************/
/******************************************************************************************/

std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  int N_GAMMA,
  double shapeGamma,
  double cov_par,
  double ex_rate) {

	// Get gamma
	std::vector<double> gammaRates = computeGammaRate(N_GAMMA, shapeGamma);
	return PyRateC_HOMPP_lik(ind, ts, te, qRate, gammaRates, cov_par, ex_rate);

}


std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double cov_par,
  double ex_rate) {

	double logDivisor = log((double)gammaRates.size());
	std::vector<double> results(fossils.size(), 0);

	for(size_t i=0; i<ind.size(); ++i) {
		size_t iF = ind[i];
		const double tl = ts[iF]-te[iF];
		double nF = fossils[iF].size();
		nF = (double)(fossils[iF].back() == 0 ? nF-1 : nF);

		if(gammaRates.size() > 1) {

			double spLogLik = 0.;
			for(size_t iG = 0; iG < gammaRates.size(); ++iG) {
				const double qGamma = gammaRates[iG]*qRate;
				const double qtl = qGamma*tl;
				const double logQ = log(qGamma);

				//lik1= -qGamma*(br_length) + log(qGamma)*k - sum(log(np.arange(1,k+1)))  -log(1-exp(-qGamma*(br_length)))
				double spGammaLogLik = -qtl + nF*logQ - logFactorialFossilCntPerSpecie[iF] - log(1.-exp(-qtl));

				if(iG == 0) spLogLik = spGammaLogLik;
				else 				spLogLik = LOG_PLUS(spLogLik,spGammaLogLik);
			}

			results[iF] = spLogLik-logDivisor; // Average the sum

		} else { // No gamma rates
			const double qtl = qRate*tl;
			const double logQ = log(qRate);
			// -q*(br_length) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(br_length)))
			results[iF] = -qtl + nF*logQ - logFactorialFossilCntPerSpecie[iF] - log(1.-exp(-qtl));
		}
	}
	return results;
}

/******************************************************************************************/
/***************************       NHPP_lik functions     *********************************/
/******************************************************************************************/

// Constants
const double C = 0.5;
const double A = 5.-(C*4.);
const double AM1 = A-1.;
const double B = 1. + (C*4.);
const double BM1 = B-1.;
const double F_BETA = boost::math::beta(A, B);
const size_t N_QUANTILE = 4;
const double LOG_N_QUANTILE = log(static_cast<double>(N_QUANTILE));
const double QUANTILE[] = {0.125, 0.375, 0.625, 0.875};

// Data augmentation
std::vector<double> processDataAugNHPP(const std::vector<int> &ind, const std::vector<double> &ts, const double qRate, const double extRate) {

	// Gamma rates
	boost::math::gamma_distribution<double> gammaDist(1., extRate);
	double sumGammaPDF = 0.;
	double GM[N_QUANTILE], gammaPDF[N_QUANTILE];
	for(size_t iQ=0; iQ<N_QUANTILE; ++iQ){
		GM[iQ] = log(1.-QUANTILE[iQ])/extRate;
		gammaPDF[iQ] = boost::math::pdf(gammaDist, -GM[iQ]);
		sumGammaPDF += gammaPDF[iQ];
	}

	// Precompute
	const double LOG_Q_RATE	= log(qRate);
	double LOG_RATIO_GAMMA_PDF[N_QUANTILE];
	for(size_t iQ=0; iQ<N_QUANTILE; ++iQ){
		LOG_RATIO_GAMMA_PDF[iQ] = log(gammaPDF[iQ]/(sumGammaPDF+1e-50));
	}

	if(sumGammaPDF <= 0.0) {
		std::vector<double> badLogLik(fossils.size(), -100000);
		return badLogLik;
	}

	std::vector<double> logLik(fossils.size(), 0.);
	for(size_t iI=0; iI<ind.size(); ++iI) {
		size_t iF = ind[iI];

		// Likelihood
		double globLik = 0.;
		for(unsigned int iX=0; iX<fossils[iF].size()-1; ++iX){
			globLik += log(pow((ts[iF]-fossils[iF][iX]), BM1));
		}
		globLik += static_cast<double>(fossils[iF].size()-1)*LOG_Q_RATE;

		// For each QUANTILE
		double locLiks[N_QUANTILE];
		for(unsigned int iQ=0; iQ<N_QUANTILE; ++iQ){
			double tl = ts[iF]-GM[iQ];
			//assert(tl != 0);
			double xB = (ts[iF])/tl; // is fossils[iF][last] always tStart-0 ?
			double intQ = boost::math::ibeta(A, B, xB) * tl * qRate;

			// Processing the equivalent of function logPERT4_density
			double F=0.;
			for(unsigned int iX=0; iX<fossils[iF].size()-1; ++iX){
				F += log(pow((fossils[iF][iX]-GM[iQ]), AM1));
			}
			F -= static_cast<double>(fossils[iF].size()-1)*log(F_BETA*pow(tl,4.));
			F += globLik;

			// Likelihood
			locLiks[iQ]  = F;															// np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1)
			locLiks[iQ] += -intQ;													// -(int_q)
			locLiks[iQ] += LOG_RATIO_GAMMA_PDF[iQ];				// + log(G_density(-GM,1,l)/den)
			locLiks[iQ] += -log(1.-exp(-intQ));						// - log(1-exp(-int_q))
		}

		double likelihood = locLiks[0];
		for(unsigned int iQ=1; iQ<N_QUANTILE; ++iQ) {
			likelihood = LOG_PLUS(likelihood, locLiks[iQ]);
		}
		likelihood -= LOG_N_QUANTILE;

		if(likelihood > 100000) likelihood = -100000;
		logLik[iF] = likelihood - logFactorialFossilCntPerSpecie[iF];
	}


	return logLik;
}


std::vector<double> processNHPPLikelihood(const std::vector<int> &ind, const std::vector<double> &ts, const std::vector<double> &te,
														 							const double qRate, const std::vector<double> &gammaRates) {

	size_t N_GAMMA = gammaRates.size();
	const double LOG_DIVISOR = log((double)N_GAMMA);

	double LOG_Q_RATE = log(qRate);
	//double LOG_Q_GAMMA_RATE[N_GAMMA];
	double* LOG_Q_GAMMA_RATE = new double[N_GAMMA];
	for(size_t iG=0; iG<N_GAMMA; ++iG) {
		LOG_Q_GAMMA_RATE[iG] = log(gammaRates[iG]*qRate);
	}

	std::vector<double> logLik(fossils.size(), 0.);
	for(size_t iI=0; iI<ind.size(); ++iI) {
		size_t iF=ind[iI];
		const double tl = ts[iF]-te[iF];

		// LOG PERT
		double globalLik=0.;
		for(size_t iX=0; iX<fossils[iF].size(); ++iX){
			globalLik += log(pow((ts[iF]-fossils[iF][iX]),BM1)*pow((fossils[iF][iX]-te[iF]), AM1));
		}
		// F -= nFossils * (log(tl^4) * fBeta(a,b)))
		globalLik -= static_cast<double>(fossils[iF].size())*log(F_BETA*pow(tl,4.));

		double spLogLik = 0.;
		if(fossils[iF].size() > 1) { // Go for gamma
			double maxTmpL = 0.;
			double sumTmpL = 0.;
			for(size_t iG=0; iG<N_GAMMA; ++iG){
				double spGammaLogLik = 0.;
				const double qGamma = gammaRates[iG]*qRate;
				const double qtl = qGamma*tl;
				double tempL = 1.-exp(-qtl);
				maxTmpL = maxTmpL < tempL ? tempL : maxTmpL;
				tempL = log(tempL);
				sumTmpL += tempL;

				spGammaLogLik = -qtl - tempL;
				spGammaLogLik += globalLik;
				spGammaLogLik += static_cast<double>(fossils[iF].size())*LOG_Q_GAMMA_RATE[iG];

				if(iG == 0) spLogLik = spGammaLogLik;
				else 				spLogLik = LOG_PLUS(spLogLik,spGammaLogLik);
			}
			spLogLik -= LOG_DIVISOR;
			double error = (maxTmpL>1.) || sumTmpL == std::numeric_limits<double>::infinity();
			if(error) spLogLik = -100000;
		} else {
			const double qtl = qRate*tl;
			spLogLik = -qtl - log(1.-exp(-qtl));
			spLogLik += globalLik;
			spLogLik += static_cast<double>(fossils[iF].size())*LOG_Q_RATE;
		}

		logLik[iF] = spLogLik - logFactorialFossilCntPerSpecie[iF];
	}

	return logLik;
}


std::vector<double> PyRateC_NHPP_lik(
	bool useDA,
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  int N_GAMMA,
  double shapeGamma,
  double covPar,
  double extRate) {

	// Get gamma
	std::vector<double> gammaRates = computeGammaRate(N_GAMMA, shapeGamma);
	return PyRateC_NHPP_lik(useDA, ind, ts, te, qRate, gammaRates, covPar, extRate);

}

// NHPP_lik with median Gamma Rates
std::vector<double> PyRateC_NHPP_lik(
	bool useDA,
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double covPar,
  double extRate) {

	std::vector<int> indNHPP, indExtent;
	for(size_t iI=0; iI<ind.size(); ++iI) {
		size_t iF = ind[iI];
		if(fossils[iF].back() == 0.) {
			indExtent.push_back(iF);
		} else {
			indNHPP.push_back(iF);
		}
	}

	// Compute for extent species
	std::vector<double> logLikExtent, logLikNHPP;
	if(useDA) {
		logLikExtent = processDataAugNHPP(indExtent, ts, qRate, extRate);
	} else {
		logLikExtent = PyRateC_HOMPP_lik(indExtent, ts, te, qRate, gammaRates, covPar, extRate);
	}

	logLikNHPP = processNHPPLikelihood(indNHPP, ts, te, qRate, gammaRates);

	// Merge results
	std::vector<double> logLik(fossils.size(), 0);
	for(size_t iL=0; iL<logLik.size(); ++iL) {
		logLik[iL] = logLikExtent[iL] + logLikNHPP[iL] ;
	}

	return logLik;
}

/******************************************************************************************/
/***********************       HPP_vec_lik functions     **********************************/
/******************************************************************************************/

// Precompute some values
void precomputeRatesAndLogRates(const std::vector<double> &qRates,
																const std::vector<double> &gammaRates,
																std::vector<double> &logGammaRates,
  															std::vector<double> &logQRates,
															  std::vector< std::vector <double> > &qGammas,
																std::vector< std::vector <double> > &logQGammas) {

	const size_t N_GAMMA = gammaRates.size();

	for(size_t iG=0; iG<N_GAMMA; iG++) logGammaRates.push_back(log(gammaRates[iG]));
	for(size_t iR=0; iR<qRates.size(); ++iR) logQRates.push_back(log(qRates[iR]));

	// PreCompute qGammas and logQGammas
	for(size_t iG=0; iG<N_GAMMA; iG++) {
		std::vector<double> qG, logQG;
		for(size_t iR=0; iR<qRates.size(); ++iR) {
			double val = gammaRates[iG]*qRates[iR];
			qG.push_back(val);
			double logVal = logGammaRates[iG]+logQRates[iR];
			logQG.push_back(logVal);
		}
		qGammas.push_back(qG);
		logQGammas.push_back(logQG);
	}
}

// Define a specie epoch span and estimated time in each epoch
void defineEpochSpanAndTime(const double start, const double end, const std::vector<double> &aEpochs,
										        std::pair<size_t, size_t> &span, std::vector<double> &timePerEpoch) {

	bool startFound = false;
	for(size_t iS=0; iS<aEpochs.size()-1; ++iS) {

		if(!startFound) { // Search for start
			if(start <= aEpochs[iS] && start >= aEpochs[iS+1]) { // Start found
				span.first = iS;
				startFound = true;
			}
		}

		if(startFound) { // Search for end
			if(end < aEpochs[iS] && end >= aEpochs[iS+1]) { // end found
				span.second = iS;
				if(timePerEpoch.empty()) { // First time, we use start
					timePerEpoch.push_back(start - end);
				} else { // Then we use whole epoch duration
					timePerEpoch.push_back(aEpochs[iS] - end); // Last time
				}

				return; // We are done
			} else {
				// Keep track of times
				if(timePerEpoch.empty()) { // First time, we use start
					timePerEpoch.push_back(start - aEpochs[iS+1]);
				} else { // Then we use whole epoch duration
					timePerEpoch.push_back(aEpochs[iS] - aEpochs[iS+1]);
				}
			}
		}
	}
}

// HPP_vec_lik using MEAN Yang discrete gamma rates
std::vector<double> PyRateC_HPP_vec_lik(std::vector <int> ind,
													 						  std::vector<double> ts,
  												 							std::vector<double> te,
																				std::vector<double> epochs,
																				std::vector<double> qRates,
																				int N_GAMMA,
																				double shapeGamma) {

	// Get gamma
	std::vector<double> gammaRates = computeGammaRate(N_GAMMA, shapeGamma);
	return PyRateC_HPP_vec_lik(ind, ts, te, epochs, qRates, gammaRates);
}

// HPP_vec_lik using MEDIAN Yang discrete gamma rates
std::vector<double> PyRateC_HPP_vec_lik(std::vector <int> ind,
																				std::vector<double> ts,
																				std::vector<double> te,
																				std::vector<double> epochs,
																				std::vector<double> qRates,
																				std::vector<double> gammaRates) {

	const size_t N_GAMMA = gammaRates.size();

	// Get logGamma and logQRates
	// This way we will required N_GAMMA + N_RATE log(...) instead of N_GAMMA*N_RATE
	std::vector<double> logGammaRates, logQRates;
	std::vector< std::vector <double> > qGammas, logQGammas;

	precomputeRatesAndLogRates(qRates, gammaRates, logGammaRates, logQRates, qGammas, logQGammas);

	// Compute the likelihood for each specie
	std::vector<double> logLik(fossils.size(), 0);
	double logDivisor = log((double)N_GAMMA);
	for(size_t iI=0; iI<ind.size(); ++iI) {
		size_t iF = ind[iI]; // Specie index in fossils
		std::pair<size_t, size_t> span;
		std::vector<double> timePerEpoch;
		defineEpochSpanAndTime(ts[iF], te[iF], epochs, span, timePerEpoch);

		size_t nFossils = fossils[iF].back() == 0 ? fossils[iF].size()-1 : fossils[iF].size();
		if(N_GAMMA > 1 && nFossils > 1) {
			double spLogLik = 0.;
			for(size_t iG = 0; iG < N_GAMMA; ++iG) {
				// For each gamma compute :
				// qGamma= YangGamma[i]*q_rates
				// lik_vec[i] = sum(-qGamma[ind]*d + log(qGamma[ind])*k_vec[ind]) - log(1-exp(sum(-qGamma[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))

				double sum1 = 0.; // sum(-qGamma[ind]*d)
				double sum2 = 0.; // sum(log(qGamma[ind])*k_vec[ind])
				// For each epoch where the specie was living
				for(size_t iE=span.first; iE<=span.second; ++iE) {
					sum1 += -qGammas[iG][iE]*timePerEpoch[iE-span.first];
					sum2 += logQGammas[iG][iE]*fossilCountPerEpoch[iF][iE];
				}

				double term1 = sum1+sum2; 														// sum(-qGamma[ind]*d + log(qGamma[ind])*k_vec[ind])
				double term2 = -log(1-exp(sum1)); 										// - log(1-exp(sum(-qGamma[ind]*d)))
				double term3 = -logFactorialFossilCntPerSpecie[iF];		// -sum(log(np.arange(1,sum(k_vec)+1)))

				double spGammaLogLik = term1+term2+term3;
				if(iG == 0) spLogLik = spGammaLogLik;
				else 				spLogLik = LOG_PLUS(spLogLik,spGammaLogLik);
			}
			logLik[iF] = spLogLik-logDivisor; // Average the sum
		} else {
			//lik = sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind]) - log(1-exp(sum(-q_rates[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))
			double sum1 = 0.; // sum(-q_rates[ind]*d)
			double sum2 = 0.; // sum(log(q_rates[ind])*k_vec[ind])
			// For each epoch where the specie was living
			for(size_t iE=span.first; iE<=span.second; ++iE) {
				sum1 += -qRates[iE]*timePerEpoch[iE-span.first];
				sum2 += logQRates[iE]*fossilCountPerEpoch[iF][iE];
			}

			double term1 = sum1+sum2; 														// sum(-qGamma[ind]*d + log(qGamma[ind])*k_vec[ind])
			double term2 = -log(1-exp(sum1)); 										// - log(1-exp(sum(-q_rates[ind]*d)))
			double term3 = -logFactorialFossilCntPerSpecie[iF];		// -sum(log(np.arange(1,sum(k_vec)+1)))
			logLik[iF] = term1+term2+term3;
		}
	}
	return logLik;
}


/***************************************************************************************************************************/

// Recursive function ? while t != 0 ?
typedef struct {
	double p, Bi;
} resultComputeP_t;

const double LOG_OF_4 = log(4.0);

resultComputeP_t computeP(int i, double t,
												  const std::vector<double> &intervalAs,
							  					const std::vector<double> &lam,
							  					const std::vector<double> &mu,
													const std::vector<double> &psi,
							  					const std::vector<double> &rho,
							  					const std::vector<double> &times) {

	resultComputeP_t res;

	if (t == 0) {
		res.p = 1.;
		res.Bi = 0.;
		return res;
	}

	double ti = times[i+1];
	double Ai = intervalAs[i];

	resultComputeP_t nextRes = computeP(i+1, ti, intervalAs, lam, mu, psi, rho, times);

	res.Bi = ((1 -2*(1-rho[i]) * nextRes.p ) * lam[i] +mu[i]+psi[i]) / Ai;
	res.p = lam[i] + mu[i] + psi[i];
	res.p -= Ai * ( ((1+res.Bi) - (1-res.Bi) * exp(-Ai*(t-ti)) ) / ((1+res.Bi) + (1-res.Bi) * exp(-Ai*(t-ti) )) );
	res.p = res.p / (2. * lam[i]);

	return res;
}

double computeQ(int i, double t,
								const std::vector<double> &intervalAs,
							  const std::vector<double> &lam,
							  const std::vector<double> &mu,
							  const std::vector<double> &psi,
							  const std::vector<double> &rho,
							  const std::vector<double> &times) {
	//std::cout << i << "  " << t << std::endl;
	resultComputeP_t res = computeP(i, t, intervalAs, lam, mu, psi, rho, times);

	double Ai_t = intervalAs[i]*(t-times[i+1]);
	double qi_t = (LOG_OF_4-Ai_t) - (2* log( exp(-Ai_t) *(1-res.Bi) + (1+res.Bi) ) );
	return qi_t;
}

double computeQt(int i, double t, double q,
								 const std::vector<double> &lam,
								 const std::vector<double> &mu,
								 const std::vector<double> &psi,
								 const std::vector<double> &times) {
	double qt = .5 * ( q - (lam[i]+mu[i]+psi[i])*(t-times[i+1]) );
	return qt;
}



//likelihood_rangeFBD
double PyRateC_FBD_T4(int nSpecies,
											std::vector<int> bint,
											std::vector<int> dint,
											std::vector<int> oint,
											std::vector<double> intervalAs,
											std::vector<double> lam,
											std::vector<double> mu,
											std::vector<double> psi,
											std::vector<double> rho,
											std::vector<double> gamma,
											std::vector<double> times,
											std::vector<double> ts,
										  std::vector<double> te,
											std::vector<double> FA) {

	/*typedef struct {size_t cntQ, cntQt;} countQ_t;
	typedef std::map< std::pair<size_t, size_t>, countQ_t > mapCount_t;

	mapCount_t  mapCount;*/

	double term4 = 0.;
	for(size_t i=0; i<nSpecies; i++) {

		double term4_q_t1 = computeQ(bint[i], ts[i], intervalAs, lam, mu, psi, rho, times);
		double term4_q_t2 = computeQ(oint[i], FA[i], intervalAs, lam, mu, psi, rho, times);
		double term4_q  = term4_q_t1 - term4_q_t2;
		//std::cout << i << " -- t4q = " <<  term4_q << std::endl;

		double term4_qt_t1_q = term4_q_t2;
		double term4_qt_t1 = computeQt(oint[i], FA[i], term4_qt_t1_q, lam, mu, psi, times);
		double term4_qt_t2_q = computeQ(dint[i], te[i], intervalAs, lam, mu, psi, rho, times);
		double term4_qt_t2 = computeQt(dint[i], te[i], term4_qt_t2_q, lam, mu, psi, times);
		double term4_qt = term4_qt_t1 - term4_qt_t2;
		//std::cout << i << " -- t4qt = " <<  term4_qt << std::endl;

		double qj_1 = 0.;
		for(int j=bint[i]; j<oint[i]; ++j) {
			qj_1 += computeQ(j+1, times[j+1], intervalAs, lam, mu, psi, rho, times);
		}
		//std::cout << i << " -- qj_1 = " <<  qj_1 << std::endl;

		double qtj_1 = 0.;
		for(int j=oint[i]; j<dint[i]; ++j) {
			double tmpQ = computeQ(j+1, times[j+1], intervalAs, lam, mu, psi, rho, times);
			qtj_1 += computeQt(j+1, times[j+1], tmpQ, lam, mu, psi, times);
		}
		//std::cout << i << " -- qtj_1 = " <<  qtj_1 << std::endl;

		double term4_qj = qj_1 + qtj_1;

		term4 += term4_q + term4_qt + term4_qj;
	}

	// log(gamma) sum
	double sumLogGamma = 0.;
	for(size_t i=0; i<gamma.size(); ++i) sumLogGamma += log(gamma[i]);
	term4 += sumLogGamma;

	return term4;
}
