export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
CUDA_VISIBLE_DEVICES=1 python run.py args/gsm_coconut.yaml