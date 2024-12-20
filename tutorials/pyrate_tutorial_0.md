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

4) **If you are using Windows**, please make sure that the path to python.exe is included in the PATH environment variables. To do so, edit the PATH environment variable and add the folder in which Python 3.x is installed (e.g. `'C:\...\Python\Python312'`) and the folder with the Python scripts (e.g. `'C:\...\Python\Python312\Scripts'`). An easy tutorial how to do that can be found for example on the [Java website](https://www.java.com/en/download/help/path.xml). 
The function -plot generates an R script that is used to produce a graphic output. The script is automatically executed by PyRate using the shell command `RScript`. If you are using Windows, please make sure that the path to `Rscript.exe` is included in the PATH environment variables (default in Mac/Linux). To do so, edit the PATH environment variable and add the \bin folder of the R installation (e.g. `'C:\Program Files\R\R-4.4.1\bin'`). An easy tutorial how to do that can be found for example on the [Java website](https://www.java.com/en/download/help/path.xml).

