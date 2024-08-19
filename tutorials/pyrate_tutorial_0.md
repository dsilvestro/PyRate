# Setting up a virtual environment for PyRate

***
 
First, make sure **Python (v.3.10 or higher)** is installed on your computer. To install or upgrade Python visit: [python.org](https://www.python.org/downloads/).    
You can run PyRate within a virtual environment to make sure all the compatible dependencies are included without affecting your system Python installation following the steps below.

1) **Create a virtual environment** typing in a terminal console (or *command prompt* on Windows): 

```
python -m venv desired_path_to_env/pyrate_env
```  
or on Windows:

```
py -m venv desired_path_to_env/pyrate_env
```  
Note that depending on the default version of Python on your machine you may have to type `python3`. 

2) **Activate the virtual environment** using on MacOS/UNIX: 

```
source desired_path_to_env/pyrate_env/bin/activate
```
or on Windows:

```
.\desired_path_to_env\pyrate_env\Scripts\activate
```  

3) **Install PyRate dependencies** in the virtual environment (on Windows use `py` instead of `python`) after replacing `your_path` with the correct path to the `PyRate-master` directory:

```
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r your_path/PyRate-master/requirements.txt
```

You can check that PyRate can now be run in your virtual environment using:
```
python your_path/PyRate-master/PyRate.py -v
```

Note: In case the message `Module FastPyRateC was not found.` appears the program will still run but use a slower implementation for some of the functions. You can follow these [instructions](https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/fastPyRateC/README.md) to install the fastPyRateC library.  

