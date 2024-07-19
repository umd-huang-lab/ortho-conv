DATASET=$1
BATCH=$2

SCALE=$3
DEPTH=$4
WIDTH=$5

KSIZE=$6
KINIT=$7

EPOCH=$8

NAME=L${SCALE}D${DEPTH}H${WIDTH}_orthoconv${KSIZE}${KINIT}

cd ../

python3 model_train.py --fresh --dataset ${DATASET} --batch_size ${BATCH} --output_dir ./outputs/${DATASET}/${NAME} --saved_checkpoint ./outputs/${DATASET}/${NAME}/glow_checkpoint_${EPOCH}.pt --num_scales ${SCALE} --num_blocks ${DEPTH} --hidden_channels ${WIDTH}  --flow_permutation orthoconv --ortho_ker_size ${KSIZE} --ortho_ker_init ${KINIT} | tee ./outputs/logs/${DATASET}_${NAME}.txt
