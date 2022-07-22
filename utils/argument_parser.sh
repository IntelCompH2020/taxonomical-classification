
ARGUMENTS=( taxonomy_name train_files dev_files test_files text_column label_column )
# Default values
taxonomy_name=$(basename "$PWD") # name of above directory (taxonomy)

display_usage() { 
    echo -en "\nUsage: $0"
	for arg in ${ARGUMENTS[@]}
	do
		echo -en " $arg=[$arg]"
	done
	echo -e "" 
} 

# check all arguments are in ARGUMENTS and parse them
if [ "$#" -gt 7 ]; then
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
		echo -e Parameter $KEY not in allowed arguments. Allowed arguments: ${ARGUMENTS[@]}
        display_usage
		exit 1
    fi
done

# Check train_files or dev_files or test_files are passed as arguments
if ([ -z "$train_files" ] && [ -z "$dev_files" ] && [ -z "$test_files" ])
then
	echo -e "No data: train_files, dev_files and test_files missing."
	display_usage
	exit 1
fi
# Check text_column and label_column are passed as arguments
if ([ -z "$text_column" ] || [ -z "$label_column" ])
then
	echo -e "No text_column or no label_column were provided"
	display_usage
	exit -1
fi

