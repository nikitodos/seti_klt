#!/bin/bash

[ -e foo.fil ] && rm foo.fil
[ -e foo_cleaned.fil ] && rm foo_cleaned.fil

unset PYTHONPATH
source /opt/conda/bin/activate klt
python --version

#python 03a_clean_filterbank.py -f ./experiment_2023-08-06/original.fil -klt -klt_win 350000 -w blimpy -var_frac 0.1

#python 03a_clean_filterbank.py -f ./experiment_2023-08-18/original.fil -klt -klt_win 300 -w blimpy -var_frac 0.8 -n klt_win_300_var_frac_08_absolut.fil
python 03a_clean_filterbank.py -f ./experiment_2023-08-18/original.fil -klt -klt_win 300 -w blimpy -var_frac 0.3 -n klt_win_300_var_frac_03_absolut.fil
#python 03a_clean_filterbank.py -f ./experiment_2023-08-18/original.fil -klt -klt_win 100000 -w blimpy -var_frac 0.8 -n klt_win_100000_var_frac_08_absolut.fil
#python 03a_clean_filterbank.py -f ./experiment_2023-08-18/original.fil -klt -klt_win 100000 -w blimpy -var_frac 0.3 -n klt_win_100000_var_frac_03_absolut.fil

