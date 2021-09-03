#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux; folder=Other;;
    Darwin*)    machine=Mac; folder=macOS;;
    CYGWIN*)    machine=Windows; folder=Windows;;
    MINGW*)     machine=Windows; folder=Windows;;
    *)          machine="UNKNOWN"
esac

echo ${folder}

if [ ${machine} == "UNKNOWN" ]; then
  echo "This type of OS is not supported. Follow the manual installation instructions."
  exit
else
  echo "The installation will proceed for a '${machine}' system."
fi

echo "############################"
echo "Preparing boost c++ library."
# Get the boost c++ library
if [ ! -d "boost" ]; then
  echo ">Downloading"
  curl -fsSL https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.zip -o boost_1_77_0.zip
  # Unzip it
  echo "> Unziping"
  unzip -q boost_1_77_0.zip
  # Move header files
  echo "> Moving files"
  mv boost_1_77_0/boost .
  # Clean up mess
  rm boost_1_77_0.zip
  rm -r boost_1_77_0
fi
echo "> done"
echo "############################"
echo ""



# Prepare swig interface
echo "############################"
echo "Preparing the python interface"
swig -c++ -python -py3 FastPyRateC.i
echo "> done"
echo "############################"
echo ""



# Compiling the c++ code
echo "############################"
echo "Compiling the c++ code and installing the library"
python3 setup.py install --install-purelib=../${folder} --install-platlib=../${folder}
myLibPath=`ls build/lib*/*.so`
myLibName=`basename ${myLibPath}`
echo $myLibName
echo "> done"
echo "############################"
echo ""



# Moving the library
echo "############################"
echo "Cleaning up."

# Cleanup
rm FastPyRateC.py
rm FastPyRateC_wrap.cxx
rm -r build
rm -r boost
echo "> done"
echo "############################"
echo ""



# Checking status
if [ -f "../${folder}/${myLibName}" ]; then

  echo " >> Successful installation of FastPyRateC (lib:${myLibName})."

else

  echo " >> An error must avec occured during the installation."
  echo " >> Try to install the library manually."

fi
