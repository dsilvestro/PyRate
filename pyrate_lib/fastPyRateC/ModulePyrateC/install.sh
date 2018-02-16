#!/bin/bash

echo "############################"
echo "Preparing boost c++ library."
# Get the boost c++ library
if [ ! -d "boost" ]; then
  echo ">Downloading"
  curl -fsSL https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.zip -o boost_1_66_0.zip
  # Unzip it
  echo "> Unziping"
  unzip -q boost_1_66_0.zip 
  # Move header files
  echo "> Moving files"
  mv boost_1_66_0/boost .
  # Clean up mess
  rm boost_1_66_0.zip
  rm -r boost_1_66_0
fi
echo "> done"
echo "############################"
echo ""

# Prepare swig interface
echo "############################"
echo "Preparing the Python interface"
swig -c++ -python FastPyRateC.i
echo "> done"
echo "############################"
echo ""


# Compiling the c++ code
echo "############################"
echo "Compiling the c++ code"
python setup.py build
echo "> done"
echo "############################"
echo ""

# Moving the library
echo "############################"
echo "Installing the library and cleaning up."
if [ ! -d "../Other/" ]; then
  echo "The 'Other' folder is missing in the parent directory. Aborting."
  exit
fi

mv build/*/_FastPyRateC.so ../Other/.

# Cleanup
rm FastPyRateC.py
rm FastPyRateC_wrap.cxx
rm -r build
echo "> done"
echo "############################"
echo ""

# Checking status

if [ -f "../Other/_FastPyRateC.so" ]; then

  echo " >> Successful installation of FastPyRateC."

else

  echo " >> An error must avec occured during the installation."
  echo " >> Try to install the library manually."

fi



