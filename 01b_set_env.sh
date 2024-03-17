#!/bin/bash

[ -e foo.fil ] && rm foo.fil
[ -e foo_cleaned.fil ] && rm foo_cleaned.fil

conda deactivate
conda deactivate
unset PYTHONPATH
source /opt/conda/bin/activate klt
python --version

pytest 01a_test_env.py

rm foo.fil

# simple runs with no changes
python 03a_clean_filterbank_your.py -f sample_your -v
python 04_compare_filterbanks.py -f1 "./data/FRB180417.fil" -f2 "./data/FRB180417_cleaned.fil"
rm ./data/FRB180417_cleaned.fil