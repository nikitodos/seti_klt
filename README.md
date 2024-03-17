# klt

## Table of contents

* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [External sources](#external-sources)

## General info

Videotutorials:

* <https://youtube.com>
Github repository:
* <https://github.com/cejkys/klt>

Set of tools tuned for SRT filterbank data

* 01_set_env.sh                     ... set environment and test it via 02_test.env.py
* 02_test_env.py                    ... pytest libraries (should pass all tests)
* 03_clean_filterbank.py            ... applies denoising
* 03_clean_filterbank_your.py       ... applies denoising using library YOUR
* 04_compare_filterbanks.py         ... returns if two filterbanks are the same
* 05_data_collector.py              ... collects all files from 2021 observation
* 06_run_cleaning.sh                ... script that runs 03_cleaning_filterbank_your.py on a given file
* 07_run_plotting.sh                ... script that runs 08_plot_data.py on a given file
* 08_plot_data.py                   ... plots waterfalls with time / frequency range specification
* 09_signal_injector.py             ... generates signal 
* 10_klt_srt_run.sh                 ... script that ran klt on 2023 internship experiment
* 11_check_hits.sh                  ... returns average number of hits on a given path & target
* 12_setigen_for_klt.sh             ... generates a set of signals into SRT data to test KLT on

## Technologies

Project is created with:

*Python                  3.11.4
*conda version :         4.10.1

libraries generated as:

``` bash
$ pip list
Package                  Version
------------------------ ---------
* anyio                    3.7.0
* argon2-cffi              21.3.0
* argon2-cffi-bindings     21.2.0
* arrow                    1.2.3
* astropy                  5.3
* asttokens                2.2.1
* attrs                    23.1.0
* backcall                 0.2.0
* beautifulsoup4           4.12.2
* bidict                   0.22.1
* bleach                   6.0.0
* blimpy                   2.1.4
* Bottleneck               1.3.7
* cffi                     1.15.1
* click                    8.1.3
* cloudpickle              2.2.1
* comm                     0.1.3
* contourpy                1.1.0
* coverage                 7.2.7
* cycler                   0.11.0
* dask                     2023.6.1
* debugpy                  1.6.7
* decorator                5.1.1
* defusedxml               0.7.1
* executing                1.2.0
* fastjsonschema           2.17.1
* fonttools                4.40.0
* fqdn                     1.5.1
* fsspec                   2023.6.0
* h5py                     3.9.0
* hdf5plugin               4.1.3
* hypothesis               6.80.0
* idna                     3.4
* imageio                  2.31.1
* iniconfig                2.0.0
* ipykernel                6.23.3
* ipython                  8.14.0
* ipython-genutils         0.2.0
* ipywidgets               8.0.6
* iqrm                     0.1.0
* isoduration              20.11.0
* jedi                     0.18.2
* Jinja2                   3.1.2
* jsonpointer              2.4
* jsonschema               4.17.3
* jupyter_client           8.3.0
* jupyter_core             5.3.1
* jupyter-events           0.6.3
* jupyter_server           2.7.0
* jupyter_server_terminals 0.4.4
* jupyterlab-pygments      0.2.2
* jupyterlab-widgets       3.0.7
* kiwisolver               1.4.4
* lazy_loader              0.3
* llvmlite                 0.40.1
* locket                   1.0.0
* markdown-it-py           3.0.0
* MarkupSafe               2.1.3
* matplotlib               3.7.1
* matplotlib-inline        0.1.6
* mdurl                    0.1.2
* mistune                  3.0.1
* nbclassic                1.0.0
* nbclient                 0.8.0
* nbconvert                7.6.0
* nbformat                 5.9.0
* nest-asyncio             1.5.6
* networkx                 3.1
* notebook                 6.5.4
* notebook_shim            0.2.3
* numba                    0.57.1
* numpy                    1.24.4
* overrides                7.3.1
* packaging                23.1
* pandas                   2.0.3
* pandocfilters            1.5.0
* parso                    0.8.3
* partd                    1.4.0
* pexpect                  4.8.0
* pickleshare              0.7.5
* Pillow                   10.0.0
* pip                      23.1.2
* platformdirs             3.8.0
* pluggy                   1.2.0
* prometheus-client        0.17.0
* prompt-toolkit           3.0.39
* psutil                   5.9.5
* ptyprocess               0.7.0
* pure-eval                0.2.2
* pycparser                2.21
* pyerfa                   2.0.0.3
* Pygments                 2.15.1
* pyparsing                2.4.7
* pyrsistent               0.19.3
* pytest                   7.4.0
* pytest-cov               4.1.0
* python-dateutil          2.8.2
* python-json-logger       2.0.7
* pytz                     2023.3
* PyWavelets               1.4.1
* PyYAML                   6.0
* pyzmq                    25.1.0
* qtconsole                5.4.3
* QtPy                     2.3.1
* rfc3339-validator        0.1.4
* rfc3986-validator        0.1.1
* rich                     13.4.2
* scikit-image             0.21.0
* scipy                    1.11.1
* Send2Trash               1.8.2
* setuptools               67.8.0
* sigpyproc                1.1.0
* six                      1.16.0
* sniffio                  1.3.0
* sortedcontainers         2.4.0
* soupsieve                2.4.1
* stack-data               0.6.2
* terminado                0.17.1
* tifffile                 2023.4.12
* tinycss2                 1.2.1
* toolz                    0.12.0
* tornado                  6.3.2
* tqdm                     4.65.0
* traitlets                5.9.0
* turbo-seti               2.3.2
* tzdata                   2023.3
* uri-template             1.3.0
* wcwidth                  0.2.6
* webcolors                1.13
* webencodings             0.5.1
* websocket-client         1.6.1
* wheel                    0.38.4
* widgetsnbextension       4.0.7
* your                     0.6.7
```

## Setup

### 01_set_env.sh

Set environment and test it.

The environment is only in the script run. If you want to set up the same environment in terminal on SRT server, run:

``` bash
unset PYTHONPATH
source /opt/conda/bin/activate klt
```

### 02_test_env.py

tests if python version and conda environment are correct
checks for correct version of numpy and matplotlib
checks functionality of scipy, numpy, astropy, your, blimpy, turboseti, setigen, 

part of 01_set_env.sh script. For separate run:

``` bash
pytest 02_test_env.py
```

Should return 0 failed, all passed, warnings can be ignored

### 03_clean_filterbank.py

To get help run:

``` bash
python 03_clean_filterbank.py -h
```

To run cleaner without cleaning (input / output are the same):

``` bash
python 03_clean_filterbank.py -f <path-to-filterbank>
```

To run cleaner using klt:

``` bash
python 03_clean_filterbank.py -f <path-to-filterbank> -klt -klt_win <klt_windows (in frequency channels)> 
```

### 09_signal_injector.py

Generates a signal into filterbank

* error: uint & float64 => adjust setigen library (setigen/setigen/frame.py at line 798) put following code:

``` python
import warnings
if not isinstance(self.data, np.float64):
    warnings.warn("Input data type differs from signal data type, conversion to be executed")
    self.data = self.data.astype(np.float64)
```

## External sources

* <https://thepetabyteproject.github.io/your/0.6.6/ipynb/Reader/>
* <https://sigproc.sourceforge.net/sigproc.pdf>
* <https://sigproc.sourceforge.net/>
* <https://docs.astropy.org/en/stable/coordinates/index.html>
* <https://blimpy.readthedocs.io/en/latest/overview.html>