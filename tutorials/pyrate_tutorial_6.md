# PyRate - Advanced settings


### Fast burnin 
##### `-fast_burnin <int>` 
Specify number of MCMC iterations using approximate Gibbs sampler for times of origination and extinction for faster convergence (default: `-fast_burnin 0`)


### Tuning parameters
##### `-multiR <boolean>` 
Set proposal mechanism for updates of speciation and extinction rates. When `-multiR 0` updates are done using sliding window, whereas `-multiR 1` sets multiplier proposals (default). Window sizes can be adjusted using `-tR`.

##### `-tT <float>` 
Window size of updates of speciation/extinction times (uniform sliding window).
Example: `-tT 1` (default)

##### `-nT` 
Maximum number of speciation/extinction times updated at a time. If set to 0, speciation/extinction times are set equal to first/last appearances and the preservation model is automatically set to HPP.
Example: `-nT 5` (default)

##### `-tQ` 
Window sizes of the preservation rate (q) and of the shape parameter of the gamma distributed rate heterogeneity (alpha), respectively (multiplier proposals).
Example: `-tQ 1.2 1.2` (default)

##### `-tR` 
Window size of updates of speciation/extinction rates (by default multiplier proposals, see also [`-multiR`](https://github.com/dsilvestro/PyRate/wiki/2.-PyRate:-command-list#-multir)).
Example: `-tR 0.05` (default)

##### `-tS` 
Window size of updates of shift times for speciation/extinction rates (uniform sliding window).
Example: `-tS 1` (default)

##### `-fS`
Frequency of updating shift times, when updating birth-death parameters (else rates are updated). The value will be automatically set to 0 when no rate shifts are being sampled or if the times of shifts are fixed (with command [`-fixSE`](https://github.com/dsilvestro/PyRate/wiki/2.-PyRate:-command-list#-fixse)).
Example: `-fS 0.7` (default)

##### `-tC` 
Window sizes of updates of correlation parameters, when using models with rates covarying with a trait. Window sizes are given for covariation with speciation, extinction, and preservation rates respectively. The parameters will be updated (or not) depending on the Covar model selected (see command [`-mCov`](https://github.com/dsilvestro/PyRate/wiki/2.-PyRate:-command-list#-mcov)).
Example: `-tC 0.025 0.025 0.1` (default)

##### `-fU` 
Update frequencies for preservation rate, birth-death parameters, and correlation parameters under the Covar model, respectively. What is left is used for updating speciation and extinction times.
Example:` -fU 0.02 0.18 0.08` (default under Covar model; updates preservation parameters with frequency 2%, birth-death parameters with frequency 18%, Covar parameters with frequency 8%, times of speciation and extinction with frequency 72%)

##### `-fR` 
Fraction of birth-death rates updated at a time (with frequency defined by the command `-fU`). This command should be used to reduce the fraction of updated rate parameters especially when running birth-death models with many shifts, e.g. defined by the command [`-fixShift`](https://github.com/dsilvestro/PyRate/wiki/2.-PyRate:-command-list#-fixshift), to improve the MCMC mixing.
Example: `-fR 1` (default)
