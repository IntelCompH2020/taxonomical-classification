#!/bin/bash

. hyperparameters.config.sh

# create run.sh
printf "#!/bin/bash" > run.sh
printf "#!/bin/bash" > launcher.sh

# create header
if [ $HPC = true ]
then
	cat ../../utils/template_hpc.sh >> launcher.sh
	taxonomy_name=$(basename "$PWD") # name of above directory (taxonomy)
	sed -i 's/<JOB_NAME>/'${taxonomy_name}'_'$(basename "$MODEL_PATH")'_'${LEARN_RATE}'/g' launcher.sh
	sed -i 's/<LOG_NAME>/'${taxonomy_name}'_'$(basename "$MODEL_PATH")'_'${LEARN_RATE}'/g' launcher.sh
	sed -i 's/<GPUS_PER_NODE>/'${GPUS_PER_NODE}'/g' launcher.sh
	sed -i 's/<CPUS_PER_NODE>/'${CPUS_PER_NODE}'/g' launcher.sh
	sed -i 's/<NUM_NODES>/'${NUM_NODES}'/g' launcher.sh
	sed -i 's/<TIME>/'${TIME}'/g' launcher.sh
else
	printf "\nsource ../../venv/bin/activate\n" >> launcher.sh
fi

# Add Argument parser
cat ../../utils/argument_parser.sh >> run.sh
cat ../../utils/argument_parser.sh >> launcher.sh

# Add hyperparameters
sed -n "/^$/,/^$/p" hyperparameters.config.sh >> run.sh
if [ $HPC = true ]
then
	sed -n "/^# SLURM$/,/^$/p" hyperparameters.config.sh >> run.sh
fi


# Add template.sh
cat ../../utils/template.sh >> run.sh
if [ $DO_EVAL != True ]
then
	sed -i 's/--evaluation_strategy steps/--evaluation_strategy no/g' run.sh
fi

# If nnodes > 1 add template
if [[ $NUM_NODES -gt 1 ]]
then
	cat ../../utils/template_multinode.sh >> run.sh
	# Add srun
	printf "srun run.sh train_files=\$train_files dev_files=\$dev_files test_files=\$test_files text_column=\$text_column label_column=\$label_column" >> launcher.sh
	printf "python -m torch.distributed.launch \$DIST_ARGS finetune_classifier.py \$MODEL_ARGS \$OUTPUT_ARGS" >> run.sh
else
	# Add bash
	printf "bash run.sh train_files=\$train_files dev_files=\$dev_files test_files=\$test_files text_column=\$text_column label_column=\$label_column" >> launcher.sh
	printf "python finetune_classifier.py \$MODEL_ARGS \$OUTPUT_ARGS" >> run.sh
fi
