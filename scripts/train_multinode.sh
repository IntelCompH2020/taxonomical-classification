#!/bin/bash

# 

ARGUMENTS=( taxonomy_name train_files dev_files test_files text_column label_column )

display_usage() { 
    echo -en "\nUsage: $0"
	for arg in ${ARGUMENTS[@]}
	do
		echo -en " $arg=[$arg]"
	done
	echo -e "" 
} 

# check all arguments are in ARGUMENTS and parse them
if [ "$#" -ne 6 ]; then
    echo -e "Illegal number of parameters"
	display_usage
	exit 1
fi
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    if echo ${ARGUMENTS[@]} | grep -w -q $KEY
    then
        KEY_LENGTH=${#KEY}
        VALUE="${ARGUMENT:$KEY_LENGTH+1}"
        export "$KEY"="$VALUE"
    else
		echo -e Parameter $KEY not in ${ARGUMENTS[@]}
        display_usage
		exit 1
    fi
done

# NAME=$1
# TRAIN_DIR=$2
# DEV_DIR=$3
# TEST_DIR=$4
# TEXT_COLUMN=$5
# LABEL_COLUMN=$6
echo $NAME



