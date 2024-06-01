DATASET=oxford_iiit_pet
METHOD=adaptformer

# DIM=32
# BIT=1
# SCALE=0.01

# DIM=1
# BIT=1
# SCALE=1.0

DIM=1
BIT=32
SCALE=1.0

# DIM=4
# BIT=32
# SCALE=1.0

LR=1e-3

### FO
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset $DATASET --method $METHOD --dim $DIM --bit $BIT --scale $SCALE
### ZO
# CUDA_VISIBLE_DEVICES=1 python main.py --ZO_Estim --lr $LR --dataset $DATASET --method $METHOD --dim $DIM --bit $BIT --scale $SCALE

CUDA_VISIBLE_DEVICES=1 nohup python main.py --ZO_Estim --lr $LR --dataset $DATASET --method $METHOD --dim $DIM --bit $BIT --scale $SCALE >/dev/null 2>&1 &

# nohup bash scripts/run.sh  >/dev/null 2>&1 &