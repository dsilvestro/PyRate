##These are the steps to compile and install the FastPyRateC library



1. Install SWIG (http://www.swig.org/download.html).
   On Linux you can use: sudo apt-get install swig 
   On MacOS you can use: brew install swig

2. In a terminal window browse to the "ModulePyrateC" directory
   (e.g. cd your_pyte_directory/pyrate_lib/fastPyRateC/ModulePyrateC)


### Automated installation (Linux/Mac):

Launch the install.sh script
Note: This script require an internet connection and might take a few minutes.


### Manual installation (all OSs):
Download the boost C++ library (https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.zip).
`wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.zip`

Unzip it inside the current folder (ModulePyRateC).
`unzip boost_1_66_0.zip`

Move the folder "boost/eigen" into the current folder
`mv boost_1_66_0/boost`

Remove the folder "boost_1_66_0/" and the "boost_1_66_0.zip" file (optional) 
`rm boost_1_66_0.zip`
`rm -r boost_1_66_0`

3.2) Create the C++/Python interface typing: 
`swig -c++ -python FastPyRateC.i`
You should have two new files in the folder (FastPyRateC.py, FastPyRateC_wrap.cxx)

3.3) Compile the library:
`python setup.py build`

3.4) Copy the library _FastPyRateC.so into the PyRate library
     folder (e.g. your_Pyrate_directory/pyrate_lib/fastPyRateC/Other/)
