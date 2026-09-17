#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux; folder=Other;;
    Darwin*)    machine=Mac; folder=macOS;;
    CYGWIN*)    machine=Windows; folder=Windows;;
    MINGW*)     machine=Windows;  folder=Windows;;
    *)          machine="UNKNOWN"
esac

if [ "${machine}" == "UNKNOWN" ]; then
  echo "This type of OS is not supported. Follow the manual installation instructions."
  exit
else
  echo "The installation will proceed for a '${machine}' system."
fi

# Determine Python command and target directory name (e.g., py313, py314)
PYTHON_BIN="${PYTHON:-python3}"
PY_VER=$(${PYTHON_BIN} -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')")
TARGET_DIR="../${folder}/${PY_VER}"

echo "Detected Python version folder: ${PY_VER}"

echo "############################"
echo "Preparing boost c++ library."
# Get the boost c++ library
if [ ! -d "boost" ]; then
  echo ">Downloading"
  curl https://github.com/boostorg/boost/archive/refs/tags/boost-1.92.0.zip -L -o boost.zip
  # Unzip it
  echo "> Unziping"
  unzip -q boost.zip
  # Move header files
  echo "> Moving files"
  mv boost-boost-* boost
  # Clean up mess
  rm boost.zip
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
${PYTHON_BIN} setup.py build
echo "> done"
echo "############################"
echo ""

# Moving the library
echo "############################"
echo "Installing the library."

# Create version-specific folder if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Remove old binary in the target folder
if [ -f "${TARGET_DIR}/_FastPyRateC.so" ]; then
  rm "${TARGET_DIR}/_FastPyRateC.so"
fi

# Move compiled extension to target version folder
mv build/*/_FastPyRateC*.so "${TARGET_DIR}/_FastPyRateC.so"

echo "> done"
echo "############################"
echo ""

# Cleanup
echo "############################"
echo "Cleaning up."
rm FastPyRateC.py
rm FastPyRateC_wrap.cxx
rm -r build
rm -r boost
echo "> done"
echo "############################"
echo ""

# Checking status
if [ -f "${TARGET_DIR}/_FastPyRateC.so" ]; then
  echo " >> Successful installation of FastPyRateC in ${TARGET_DIR}."
else
  echo " >> An error must have occurred during the installation."
  echo " >> Try to install the library manually."
fi
