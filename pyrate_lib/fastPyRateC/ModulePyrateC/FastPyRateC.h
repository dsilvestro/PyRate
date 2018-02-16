#ifndef PYRATEC_H_
#define PYRATEC_H_

#include <math.h> 
#include <vector>
#include <string>

// init function -- MUST BE RAN PRIOR TO ANY CALL TO ANY OTHER FUNCTIONS
// 1) Set fossils
void PyRateC_setFossils(std::vector< std::vector <double> > aFossils);
// 2) Set epochs if there are some
void PyRateC_initEpochs(std::vector< double > aEpochs);

// Gamma distribution pdf
double PyRateC_getLogGammaPDF(double values, double shape, double scale);
std::vector<double> PyRateC_getLogGammaPDF(std::vector<double> values, double shape, double scale);

// BD Partial likelihood
std::vector<double> PyRateC_BD_partial_lik(
  std::vector<double> ts, 
  std::vector<double> te, 
  std::vector<double> timeFrameL,
  std::vector<double> timeFrameM,
  std::vector<double> ratesL,
  std::vector<double> ratesM);

// HOMPP_lik with mean Gamma Rates
std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te,
	double qRate,
  int N_GAMMA,
  double shapeGamma,
  double cov_par,
  double ex_rate);

// HOMPP_lik with median Gamma Rates
std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double cov_par,
  double ex_rate);

// NHPP_lik with mean Gamma Rates
std::vector<double> PyRateC_NHPP_lik(
	bool useDA,
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te,
	double qRate,
  int N_GAMMA,
  double shapeGamma,
  double cov_par,
  double ex_rate);

// NHPP_lik with median Gamma Rates
std::vector<double> PyRateC_NHPP_lik(
	bool useDA,
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double cov_par,
  double ex_rate);

// HPP_vec_lik with mean Gamma Rates
std::vector<double> PyRateC_HPP_vec_lik(
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te, 
	std::vector<double> epochs,
	std::vector<double> qRates,
  int N_GAMMA,
  double shapeGamma);

// HPP_vec_lik with median Gamma Rates
std::vector<double> PyRateC_HPP_vec_lik(
	std::vector <int> ind, 
	std::vector<double> ts, 
  std::vector<double> te, 
	std::vector<double> epochs,
	std::vector<double> qRates,
  std::vector<double> gammaRates);


#endif /* PYRATEC_H_ */
