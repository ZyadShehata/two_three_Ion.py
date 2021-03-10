# ParallelTemporalCorrelation #
## About ##
This is a small Python script that uses parallel computing to calculate correlation functions from timetag files of the double LinCam for Ion Experiments.<br>
Written by Stefan Richter `<stefan.richter@physical-perception.de>`

## Installation & Updating ##

This program is intend for use with Windows only!
It depends on python 3.6. THe best way is to create a new virtual environment with anaconda with the following packages:
```
- numpy
- matplotlib
- scipy
- numba
- PyQt5
- qdarkstyle
```

It can be achieved by running these commands:
```
conda create -n <env_name> -python=3.6
```
and afterwards:
```
pip install numpy matplotlib scipy numba PyQt5 qdarkstyle
```

## Usage ##
To use the program, just switch to your python3.6 env and let it run
```
source activate <env_name>
python main.py
```


