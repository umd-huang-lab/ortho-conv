DATASET=$1
BATCH=$2

SCALE=$3
DEPTH=$4
WIDTH=$5

FPERM=$6

EPOCH=$7

NAME=L${SCALE}D${DEPTH}H${WIDTH}_${FPERM}

cd ../

python3 model_train.py --fresh --dataset ${DATASET} --batch_size ${BATCH} --output_dir ./outputs/${DATASET}/${NAME} --saved_checkpoint ./outputs/${DATASET}/${NAME}/glow_checkpoint_${EPOCH}.pt --num_scales ${SCALE} --num_blocks ${DEPTH} --hidden_channels ${WIDTH}  --flow_permutation ${FPERM} | tee ./outputs/logs/${DATASET}_${NAME}.txt