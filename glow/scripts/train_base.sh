DATASET=$1
BATCH=$2

SCALE=$3
DEPTH=$4
WIDTH=$5

FPERM=$6

NAME=L${SCALE}D${DEPTH}H${WIDTH}_${FPERM}

cd ../

python3 model_train.py --fresh --dataset ${DATASET} --download --batch_size ${BATCH} --output_dir ./outputs/${DATASET}/${NAME} --num_scales ${SCALE} --num_blocks ${DEPTH} --hidden_channels ${WIDTH}  --flow_permutation ${FPERM} | tee ./outputs/logs/${DATASET}_${NAME}.txt