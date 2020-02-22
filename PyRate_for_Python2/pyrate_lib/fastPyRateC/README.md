## These are the steps to compile and install the FastPyRateC library


1. Install SWIG (http://www.swig.org/download.html).
   On Linux you can use: `sudo apt-get install swig`
   On MacOS you can use: `brew install swig`

2. Install the curl command if not available.
    On Linux you can use: `sudo apt-get install curl`
    On MacOS you can use: `brew install curl`

3. In a terminal window browse to the "ModulePyrateC" directory
   `e.g. cd your_PyRate_directory/pyrate_lib/fastPyRateC/ModulePyrateC`


### Automated installation (Linux/Mac):

Launch the install script
`bash install.sh`

Note:
1. This script require an internet connection and might take a few minutes.
2.  It should also work for Cygwin and Mingw but has not been tested.


### Manual installation (all OSs):
* Download the boost C++ library (https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.zip).
`curl -fsSL https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.zip -o boost_1_66_0.zip`

* Unzip it inside the current folder (ModulePyRateC).
`unzip boost_1_66_0.zip`

* Move the folder "boost_1_66_0/boost" into the current folder
`mv boost_1_66_0/boost`

* Remove the folder "boost_1_66_0/" and the "boost_1_66_0.zip" file (optional)
`rm boost_1_66_0.zip`
`rm -r boost_1_66_0`

* Create the C++/Python interface typing:
`swig -c++ -python FastPyRateC.i`
You should have two new files in the folder (FastPyRateC.py, FastPyRateC_wrap.cxx)

* Compile the library:
`python setup.py build`

* Remove the files that are no longer required
`rm -r boost FastPyRateC.py FastPyRateC_wrap.cxx`

* Copy the library _FastPyRateC.so into the PyRate library folder corresponding to your OS
`e.g. your_Pyrate_directory/pyrate_lib/fastPyRateC/Other/`
Folder `Other` for any kind of Linux OS. `macOS` and `Windows` folders for their respective OS system.
