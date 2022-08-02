
MODEL_NAME=$( basename $MODEL_PATH )
BATCH_SIZE_PER_GPU=$(( $TRAIN_BATCH_SIZE*$TRAIN_ACC_STEPS ))
TOTAL_BATCH_SIZE=$(( $BATCH_SIZE_PER_GPU*$NUM_NODES*$GPUS_PER_NODE ))
DIR_NAME=${MODEL_NAME}_${taxonomy_name}_${NUM_EPOCHS}_${TOTAL_BATCH_SIZE}_${LEARN_RATE}_${WEIGHT_DECAY}_${WARMUP}_$(date +"%m-%d-%y_%H-%M")

export HF_HOME=$CACHE_DIR/$DIR_NAME/huggingface

MODEL_ARGS=" \
 --model_name_or_path $MODEL_PATH \
 --data_loading_script $DATA_LOADER_PATH \
 --train_set_path $train_files \
 --dev_set_path $dev_files \
 --test_set_path $test_files \
 --text_column $text_column \
 --label_column $label_column \
 --problem_type $PROBLEM_TYPE \
 --do_train $DO_TRAIN \
 --do_eval $DO_EVAL \
 --do_predict $DO_PREDICT \
 --num_train_epochs $NUM_EPOCHS \
 --max_seq_length $MAX_SEQ_LENGTH \
 --pad_to_max_length $PAD_TO_MAX_LEN \
 --gradient_accumulation_steps $TRAIN_ACC_STEPS \
 --eval_accumulation_steps $EVAL_ACC_STEPS \
 --per_device_train_batch_size $TRAIN_BATCH_SIZE \
 --per_device_eval_batch_size $EVAL_BATCH_SIZE \
 --learning_rate $LEARN_RATE \
 --warmup_ratio $WARMUP \
 --weight_decay $WEIGHT_DECAY \
 --metric_for_best_model f1 \
 --evaluation_strategy steps \
 --save_strategy steps \
 --save_steps $SAVE_STEPS \
 --ddp_find_unused_parameters $DDP_FIND_UNUSED_PARAMETERS \
 --seed $SEED \
 "

OUTPUT_ARGS=" \
 --output_dir $OUTPUT_DIR/$DIR_NAME \
 --overwrite_output_dir \
 --logging_dir $LOGGING_DIR/$DIR_NAME \
 --logging_strategy steps \
 --logging_steps $LOGGING_STEPS \
 --cache_dir $CACHE_DIR/$DIR_NAME \
 --overwrite_cache $OVERWRITE_CACHE \
 --load_best_model_at_end \
 "

