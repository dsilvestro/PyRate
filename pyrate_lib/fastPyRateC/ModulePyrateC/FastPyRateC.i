%module FastPyRateC

%{
#define SWIG_FILE_WITH_INIT
#include "FastPyRateC.h"
%}

%include "std_vector.i"
%include "std_string.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
 	 %template(vector2d) vector< vector<double> >;
};

void PyRateC_setFossils(std::vector< std::vector <double> > aFossils);
void PyRateC_initEpochs(std::vector< double > aEpochs);

double PyRateC_getLogGammaPDF(double values, double shape, double scale);
std::vector<double> PyRateC_getLogGammaPDF(std::vector<double> values, double shape, double scale);

std::vector<double> PyRateC_BD_partial_lik(
  std::vector<double> ts,
  std::vector<double> te,
  std::vector<double> timeFrameL,
  std::vector<double> timeFrameM,
  std::vector<double> ratesL,
  std::vector<double> ratesM);

std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  int N_GAMMA,
  double shapeGamma,
  double cov_par,
  double ex_rate);

std::vector<double> PyRateC_HOMPP_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double cov_par,
  double ex_rate);

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

std::vector<double> PyRateC_NHPP_lik(
	bool useDA,
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	double qRate,
  std::vector<double> gammaRates,
  double cov_par,
  double ex_rate);

std::vector<double> PyRateC_HPP_vec_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	std::vector<double> epochs,
	std::vector<double> qRates,
  int N_GAMMA,
  double shapeGamma);

std::vector<double> PyRateC_HPP_vec_lik(
	std::vector <int> ind,
	std::vector<double> ts,
  std::vector<double> te,
	std::vector<double> epochs,
	std::vector<double> qRates,
  std::vector<double> gammaRates);

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
  std::vector<double> FA);
