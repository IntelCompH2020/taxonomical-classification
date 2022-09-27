#!/bin/bash

if uname -a | grep -q amd
then
	bash /gpfs/scratch/bsc88/bsc88148/amd_setup/setup.sh -n venv
else
	python3 -m venv venv
	source venv/bin/activate
	pip install -r utils/requirements.txt
fi
