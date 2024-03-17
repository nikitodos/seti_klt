#!/bin/bash

unset PYTHONPATH
source /opt/conda/bin/activate klt
export CUPY_ACCELARATORS=cutensor
export HDF5_USE_FILE_LOCKING='FALSE'
python --version

File=$1
Lines=$(cat $File)

timestamp=`date "+%d-%m-%y_%H-%M"`
mkdir /home/obs/klt/results/$timestamp
cd /home/obs/klt/results/$timestamp

for Line in $Lines
do
   /home/obs/.local/bin/turboSETI -M 4 --blank_dc y -s 10 -g y $Line
done
