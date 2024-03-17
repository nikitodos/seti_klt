#!/bin/bash

PATH_TO_OBSERVATION=$1
TARGET=$2
HEADER_LENGTH=9
SUMMARY=0

#after remove_spikes
#for FILE in `ls $PATH_TO_OBSERVATION | grep $TARGET | grep ".dat" | grep "NO_DC"`
#before remove spikes
for FILE in `ls $PATH_TO_OBSERVATION | grep "$TARGET" | grep ".dat"`
do
    PATH=$PATH_TO_OBSERVATION/$FILE
    echo $PATH
    TOTAL_LINES=`/bin/cat $PATH | /usr/bin/wc -l`
    HITS=$(( $TOTAL_LINES - $HEADER_LENGTH ))
    echo $HITS
    SUMMARY=$(( $SUMMARY + $HITS)) 
done

#after remove_spikes
#NUMBER_OF_FILES=`/bin/ls $PATH_TO_OBSERVATION | /bin/grep $TARGET | /bin/grep ".dat" | /bin/grep "NO_DC" | /usr/bin/wc -l`
#before remove spikes
NUMBER_OF_FILES=`/bin/ls $PATH_TO_OBSERVATION | /bin/grep $TARGET | /bin/grep ".dat" | /usr/bin/wc -l`

AVERAGE_HITS=$(( $SUMMARY / $NUMBER_OF_FILES))
echo "AVERAGE_HITS FOR TARGET $TARGET:"
echo $AVERAGE_HITS  