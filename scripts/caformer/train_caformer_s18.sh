DATA_PATH=/disk2/xrh/datasets/imagenet1k  # modify data path here
CODE_PATH=./


ALL_BATCH_SIZE=4096
NUM_GPU=4
GRAD_ACCUM_STEPS=16 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MASTER_PORT=29501

cd $CODE_PATH && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
--nproc_per_node=$NUM_GPU \
--master_port=$MASTER_PORT \
train.py --data-dir $DATA_PATH \
--model caformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 \
> log/caformer_s18.log 2>&1