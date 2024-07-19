DATASET=$1
BATCH=$2

SCALE=$3
DEPTH=$4
WIDTH=$5

KSIZE=$6
KINIT=$7

NAME=L${SCALE}D${DEPTH}H${WIDTH}_orthoconv${KSIZE}${KINIT}

cd ../

python3 model_train.py --fresh --dataset ${DATASET} --download --batch_size ${BATCH} --output_dir ./outputs/${DATASET}/${NAME} --num_scales ${SCALE} --num_blocks ${DEPTH} --hidden_channels ${WIDTH} --flow_permutation orthoconv --ortho_ker_size ${KSIZE} --ortho_ker_init ${KINIT} | tee ./outputs/logs/${DATASET}_${NAME}.txt