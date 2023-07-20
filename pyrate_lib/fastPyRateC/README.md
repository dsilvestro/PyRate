## These are the steps to compile and install the FastPyRateC library


1. Install SWIG (http://www.swig.org/download.html).
   On (Debian based) Linux you can use: `sudo apt-get install swig`
   On MacOS you can use: `brew install swig`

2. Install the curl command if not available.
    On (Debian based) Linux you can use: `sudo apt-get install curl`
    On MacOS you can use: `brew install curl`

3. In a terminal window browse to the "ModulePyrateC" directory
   `e.g. cd your_PyRate_directory/pyrate_lib/fastPyRateC/ModulePyrateC`


### Automated installation (Linux/Mac):

Launch the install script
`bash install.sh`

Note:
1. This script require an internet connection and might take a few minutes.
2. It should also work for Cygwin and Mingw but has not been tested.


### Manual installation (all OSs):
* Download the boost C++ library [https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.zip](https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.zip).
`curl https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.zip -L -o boost_1_82_0.zip`

* Unzip it inside the current folder (ModulePyRateC).
`unzip boost_1_82_0.zip`

* Move the folder "boost_1_82_0/boost" into the current folder
`mv boost_1_82_0/boost .`

* Remove the folder "boost_1_82_0/" and the "boost_1_82_0.zip" file (optional)
`rm boost_1_82_0.zip`
`rm -r boost_1_82_0`

* Create the C++/Python interface typing:
`swig -c++ -python FastPyRateC.i`
You should have two new files in the folder (FastPyRateC.py, FastPyRateC_wrap.cxx)

* Compile the library:
`python setup.py build`

* Remove the files that are no longer required
`rm -r boost FastPyRateC.py FastPyRateC_wrap.cxx`
`rm -r build`

* Rename the library inside .../pyrate_lib/fastPyRateC/lib.(name according to your OS system) from e.g "_FastPyRateC.cpython-311-x86_64-linux-gnu.so" or "_FastPyRateC.cpython-311-darwin.so" to "_FastPyRateC.so" and copy it into the PyRate library folder corresponding to your OS
`e.g. your_Pyrate_directory/pyrate_lib/fastPyRateC/Other/`
Folder `Other` for any kind of Linux OS. `macOS` and `Windows` folders for their respective OS system.
