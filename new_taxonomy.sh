#!/bin/bash

# List of arguments
ARGUMENTS=( taxonomy_name )

# Function to display usage information
display_usage() { 
    echo -en "\nUsage: $0"
	for arg in ${ARGUMENTS[@]}
	do
		echo -en " $arg=[$arg]"
	done
	echo -e "" 
}

# Ensure that only one argument is passed
if [ "$#" -ne 1 ]; then
    echo -e "Illegal number of parameters"
	display_usage
	exit 1
fi

# Check that all arguments were provided and parse them
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

# Create a working directory with all the necessary files and subdirectories.
taxonomy_dir=taxonomies/$taxonomy_name
if [ -d $taxonomy_dir ]
then
	echo "A directory for the given taxonomy ($taxonomy_name) already exists."
	display_usage
	exit 1
else
	mkdir $taxonomy_dir
	mkdir $taxonomy_dir/output
	mkdir $taxonomy_dir/tb
	mkdir $taxonomy_dir/logs
	cp utils/hyperparameters.config.sh $taxonomy_dir/hyperparameters.config.sh
	cp utils/generate_run.sh $taxonomy_dir/generate_run.sh
	cp scripts/finetune_classifier.py $taxonomy_dir/finetune_classifier.py
	cp scripts/data_loader.py $taxonomy_dir/data_loader.py
fi
