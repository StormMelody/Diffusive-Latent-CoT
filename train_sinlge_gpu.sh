CUDA_VISIBLE_DEVICES=1 python /data2/xxw_data/projects/LLM/Diffusive-Latent-CoT/training/strategies/train_coconut_batch.py \
    --epochs 1 --batch_size 32 --learning_rate 0.0001 --debug
    # --resume_from_checkpoint /data2/xxw_data/projects/LLM/Diffusive-Latent-CoT/checkpoints/step-000001-epoch-01-loss=348.4400.pt