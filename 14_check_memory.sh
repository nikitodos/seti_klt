#/bin/bash

TOTAL=`free | grep 'Mem' | awk '{print $2}'`
OCCUPIED=`free | grep 'Mem' | awk '{print $3}'`
FREE=`free | grep 'Mem' | awk '{print $4}'`

TOTAL_GB=$(( $TOTAL / 1000000 ))
OCCUPIED_GB=$(( $OCCUPIED / 1000000 ))
FREE_GB=$(( $FREE / 1000000 ))


OCCUPATION_PERCENTAGE=$((100 * $OCCUPIED_GB / $TOTAL_GB))
FREE_PERCENTAGE=$((100 * $FREE_GB / $TOTAL_GB ))

echo "$OCCUPIED_GB GB / $TOTAL_GB GB used ($OCCUPATION_PERCENTAGE%)"
echo "$FREE_GB GB / $TOTAL_GB GB free ($FREE_PERCENTAGE%)" 