#!/bin/bash

[ -e foo.fil ] && rm foo.fil
[ -e foo_cleaned.fil ] && rm foo_cleaned.fil

unset PYTHONPATH
source /opt/conda/bin/activate klt
python --version

File=$1
Lines=$(cat $File)

for Line in $Lines
do
    python 03a_clean_filterbank.py -f $Line -o data/ -v -klt -klt_win 512 | tee ./klt_run.log
    [ -e foo.fil ] && rm foo.fil
    [ -e foo_cleaned.fil ] && rm foo_cleaned.fil
    echo "KLT DONE ON $Line"
done

