#!/bin/bash

ARGUMENTS=( taxonomy_name train_files dev_files test_files text_column label_column config_file )
# Default values
taxonomy_name=$(basename "$PWD") # name of above directory (taxonomy)
config_file=hyperparameters.config

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
		echo -e Parameter $KEY not in ${ARGUMENTS[@]}
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

# read config
. $config_file


# source venv and module load
if uname -a | grep -q amd
then
	module load cmake/3.18.2 gcc/10.2.0 rocm/5.1.1 mkl/2018.4 intel/2018.4 python/3.7.4
	source ../../venv/bin/activate
	export LD_LIBRARY_PATH=../../external-lib:$LD_LIBRARY_PATH
elif uname -a | grep -q p9
then
	module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 \
					atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 \
					python/3.7.4_ML arrow/3.0.0 text-mining/2.0.0 torch/1.9.0a0 torchvision/0.11.0
else
	source ../../venv/bin/activate
fi
