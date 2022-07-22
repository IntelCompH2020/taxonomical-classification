
export GPUS_PER_NODE=$GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR
export MASTER_PORT=$MASTER_PORT

DIST_ARGS=" \
 --nproc_per_node=$GPUS_PER_NODE \
 --nnodes=$NNODES \
 --rdzv_id=$SLURM_JOB_ID \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
 "

