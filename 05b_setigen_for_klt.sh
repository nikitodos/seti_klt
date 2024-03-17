#!/bin/bash

unset PYTHONPATH
source /opt/conda/bin/activate klt
python --version

python 05a_signal_injector.py -v -w blimpy -f ./data/FRB180417.fil -o ./data/injected -stg -snr 200 -df 30000000
python 05a_signal_injector.py -v -w blimpy -f ./data/FRB180417.fil -o ./data/injected -tls -snr 200 -df 30000000
