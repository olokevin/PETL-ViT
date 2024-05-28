DATASET=oxford_iiit_pet
METHOD=adaptformer
# METHOD=adaptformer-bihead

CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method $METHOD --dim 32 --bit 1 --load_config --model_path './ckpts'

# CUDA_VISIBLE_DEVICES=0 nohup python main.py --dataset $DATASET --method $METHOD --dim 32 --bit 1 --load_config --model_path './ckpts' >/dev/null 2>&1 &