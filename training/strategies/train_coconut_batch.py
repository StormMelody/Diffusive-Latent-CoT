import json
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from pathlib import Path
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import transformers
from models import DiffusiveCoT
from training.metrics import VLAMetrics
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from overwatch import initialize_overwatch
import wandb
from datetime import datetime
from typing import Optional
import torch.nn.utils.rnn as rnn_utils
from transformers.models.gpt2 import GPT2LMHeadModel

class GSM8KDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_length=512, embedding_model=None, debug=False, debug_samples=4):
        """
        初始化数据集
        :param data: 可以是包含样本的列表，或JSON文件的路径
        :param tokenizer: 用于文本编码的tokenizer（如Hugging Face的tokenizer）
        :param max_length: 输入序列的最大长度
        :param embedding_model: 用于生成嵌入的模型
        :param debug: 是否使用调试模式（只使用少量样本）
        :param debug_samples: 调试模式下使用的样本数量
        """
        if isinstance(data, str):
            # 如果输入是文件路径，则加载JSON文件
            with open(data, 'r') as f:
                self.data = json.load(f)
        else:
            # 直接使用传入的列表
            self.data = data
        if debug:
            self.data = self.data[:debug_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_model = embedding_model
        self.device = next(embedding_model.parameters()).device if embedding_model is not None else 'cpu'
        
        # 预处理和过滤数据，移除空steps的样本
        self._preprocess_data()
        
        # 缓存，用于存储已计算的嵌入
        self.embedding_cache = {}

    def _preprocess_data(self):
        """
        预处理数据，移除无效样本
        """
        valid_data = []
        for idx, sample in enumerate(self.data):
            if len(sample.get('steps', [])) > 0:
                valid_data.append(sample)
            else:
                print(f"Warning: Removing sample with empty steps at index {idx}")
                print(f"Question: {sample.get('question', 'N/A')}")
                print(f"Answer: {sample.get('answer', 'N/A')}")
        
        self.data = valid_data
        print(f"Preprocessed dataset contains {len(self.data)} valid samples")

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)

    def _get_embedding(self, text, is_step=False):
        """
        获取文本的嵌入向量，使用缓存避免重复计算
        """
        cache_key = f"{text}_{is_step}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # 将数据移动到模型所在的设备
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():  # 不需要梯度计算
                outputs = self.embedding_model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1]  # 使用最后一层的hidden_states
                
                if is_step:
                    # 对于step，我们需要squeeze并取平均
                    embedding = embedding.squeeze(0).mean(dim=0)  # [768]
                else:
                    # 对于question，我们需要取平均
                    embedding = embedding.mean(dim=1)  # [1, 768]
            
            # 存入缓存
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            print(f"Text content: '{text}'")
            return None

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        """
        # 如果索引超出范围，返回第一个样本
        if idx >= len(self.data):
            idx = 0
            
        sample = self.data[idx]
        
        # 提取原始文本字段
        question = sample['question']
        answer = sample['answer']
        
        # 处理steps
        original_steps = ['<BOD>'] + sample['steps'] + ['<EOD>']
        
        # 获取question的嵌入
        question_embedding = self._get_embedding(question)
        if question_embedding is None:
            # 如果获取嵌入失败，尝试下一个样本
            return self.__getitem__((idx + 1) % len(self.data))
        
        # 处理answer的token ids
        tokenizer_output = self.tokenizer(answer + self.tokenizer.eos_token, padding=True, truncation=True, return_tensors='pt')
        answer_token_ids = tokenizer_output.input_ids
        
        # 处理steps的嵌入
        steps_embeddings = []
        for step in original_steps:
            if not step:
                continue
                
            step_embedding = self._get_embedding(step, is_step=True)
            if step_embedding is not None:
                steps_embeddings.append(step_embedding)
        
        # 确保至少有一个步骤
        if len(steps_embeddings) == 0:
            print(f"Warning: No valid steps found for item {idx}")
            return self.__getitem__((idx + 1) % len(self.data))
        
        steps_tensor = torch.stack(steps_embeddings, dim=0)  # [num_steps, 768]
        
        # 返回处理后的结果
        return question_embedding, answer_token_ids, steps_tensor

    def collate_fn(batch):
        """
        自定义的collate函数，用于DataLoader
        """
        questions, answers, steps_list = zip(*batch)
        
        # Stack questions (these should have consistent shapes)
        questions = torch.cat(questions, dim=0)
        
        # Process answers
        answers = [tensor.squeeze(0) for tensor in answers]
        answers_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=50256)
        answers_label_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=-100)
        
        # Handle variable-length steps using padding
        steps_padded = rnn_utils.pad_sequence(steps_list, batch_first=True, padding_value=0)
        
        return questions, answers_padded, steps_padded, answers_label_padded

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = None # Will be initialized in main() with distributed parameters

def setup_distributed(rank, world_size):
    """Sets up the distributed training environment."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleans up the distributed training environment."""
    dist.destroy_process_group()

def main(
    local_rank: int, # Added for DDP
    CoTModel: DiffusiveCoT,
    dataset: GSM8KDataset,
    save_interval: int = 2500,
    save_full_model: bool = True,
    epochs: int = 1,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    batch_size=1,
    max_steps: Optional[int] = None,
    enable_mixed_precision_training: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    world_size: int = 1, # Added for DDP
    resume_from: Optional[str] = None # 添加恢复训练的参数
    ) -> None:

    if world_size > 1:
        setup_distributed(local_rank, world_size)

    global overwatch
    overwatch = initialize_overwatch("train_coconut_batch")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=GSM8KDataset.collate_fn,
        shuffle=False if sampler else True, # Shuffle is handled by DistributedSampler
        num_workers=4, # Adjust as needed
        pin_memory=True # Recommended for DDP
    )
    # collate_fn=GSM8KDataset.collate_fn
    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import get_cosine_schedule_with_warmup  # 添加这一行
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    CoTModel = CoTModel.to(device)
    if world_size > 1:
        CoTModel = DDP(CoTModel, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = AdamW(CoTModel.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.99
    )
    print(f"Learning rate in optimizer: {optimizer.param_groups[0]['lr']}")
    
    # 初始化全局步数和起始epoch
    global_step = 0
    start_epoch = 0
    
    # 计算训练步数
    num_training_steps = (len(dataset) * epochs) // (batch_size * grad_accumulation_steps)
    # 设置warmup步数，通常为总训练步数的10%
    num_warmup_steps = int(0.1 * num_training_steps)
    
    # 使用带有warmup的余弦学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 加载检查点进行续训（如果指定）
    if resume_from is not None:
        if os.path.isfile(resume_from):
            if overwatch.is_rank_zero():
                overwatch.info(f"Loading checkpoint from {resume_from}")
            
            # 加载检查点
            checkpoint = torch.load(resume_from, map_location=device)
            
            # 恢复模型权重
            if isinstance(CoTModel, DDP):
                CoTModel.module.load_state_dict(checkpoint["model"])
            else:
                CoTModel.load_state_dict(checkpoint["model"])
            
            # 恢复全局步数和epoch
            global_step = checkpoint.get("global_step", 0)
            start_epoch = checkpoint.get("epoch", 0)
            
            # 恢复优化器状态（如果有）
            if "optimizer" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                
            # 恢复学习率调度器状态（如果有）
            if "scheduler" in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint["scheduler"])
                
            if overwatch.is_rank_zero():
                overwatch.info(f"Resumed training from step {global_step}, epoch {start_epoch}")
        else:
            if overwatch.is_rank_zero():
                overwatch.warning(f"Checkpoint file {resume_from} not found. Starting training from scratch.")
    
    # 初始化wandb
    if overwatch.is_rank_zero():  # 只在主进程初始化wandb
        wandb.init(
            mode="offline",
            project="diffusive-cot",  # 项目名称
            name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}",  # 运行名称
            config={
                "batch_size": batch_size,
                "grad_accumulation_steps": grad_accumulation_steps,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "resumed_from": resume_from if resume_from else "None"
            }
        )
    # === Train ===
    status = "Training DiffusiveCoT"
    sequence_length = 1024
    
    # 计算目标步数
    if world_size > 1:
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    else:
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    target_steps = epochs * target_steps_per_epoch

    # num_training_steps should be calculated based on the total number of optimizer steps
    num_training_steps = target_steps # This seems more direct

    # Re-initialize lr_scheduler with the potentially updated num_training_steps
    # 如果没有从检查点恢复学习率调度器，则重新初始化
    if resume_from is None or "scheduler" not in checkpoint:
        num_warmup_steps = int(0.1 * num_training_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    # 计算剩余步数用于tqdm进度条
    remaining_steps = target_steps - global_step
    
    with tqdm(
        total=remaining_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        CoTModel.train()
        optimizer.zero_grad()
        
        # 从指定的epoch开始训练
        for epoch_idx in range(start_epoch, epochs):
            if overwatch.is_rank_zero():
                overwatch.info(f"Starting Epoch {epoch_idx + 1}/{epochs}")
            
            if world_size > 1 and hasattr(dataloader.sampler, "set_epoch") and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch_idx)

            for train_idx, batch in enumerate(dataloader):
                print("train_idx is: ", train_idx)
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                question, answer, steps, answer_label = batch     # shape: [4, 1, 768] [4, 1, 768] [4, 4, 768]
                # import pdb; pdb.set_trace()
                # device = next(CoTModel.parameters()).device # Device is already set
                question = question.to(device, non_blocking=True)
                answer = answer.to(device, non_blocking=True)
                steps = steps.to(device, non_blocking=True)
                answer_label = answer_label.to(device, non_blocking=True)
                with torch.autocast(
                    "cuda", dtype=mixed_precision_dtype, enabled=enable_mixed_precision_training
                ):
                    loss = CoTModel(
                        question=question,
                        steps=steps,
                        answer=answer,
                        answer_label=answer_label,
                        output_hidden_states = True,
                    )

                # Commit Loss =>> Backward!
            
                # metrics.commit(loss=loss)
                
                normalized_loss = loss / grad_accumulation_steps
                normalized_loss.backward()

                # === Gradient Step ===
                # Step =>> Only if Done w/ Gradient Accumulation
                if (train_idx + 1) % grad_accumulation_steps == 0:
                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                    # clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    # epoch_for_logging is the original 'epoch' calculation based on global_step
                    epoch_for_logging = global_step // (len(dataloader) // grad_accumulation_steps) if (len(dataloader) // grad_accumulation_steps) > 0 else 0
                    
                    # 直接使用wandb记录
                    if overwatch.is_rank_zero():
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch_for_logging, # Using original epoch calculation for wandb
                            "train/global_step": global_step,
                        }, step=global_step)
                        
                        # 打印训练信息 - updated to show current epoch_idx from the new loop
                        print(f"Step {global_step}/{target_steps}, Epoch {epoch_idx + 1}/{epochs} (LogEpoch: {epoch_for_logging}), Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                    
                    # 保存checkpoint (using epoch_for_logging for consistency with original naming)
                    # Save checkpoint logic needs to be rank-aware
                    # For example, save every N steps or at the end of epochs
                    # Let's simplify to save at specific global_step milestones, only on rank 0
                    # The original logic for saving 5 times based on target_steps * i // 5 is fine.
                    if overwatch.is_rank_zero():
                        for i in range(1, 6): # Save 5 times during training
                            if global_step == target_steps * i // 5:
                                checkpoint_dir = Path("./checkpoints")
                                checkpoint_dir.mkdir(exist_ok=True)
                                
                                train_loss = loss.item()
                                # Checkpoint name uses epoch_for_logging
                                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch_for_logging:02d}-loss={train_loss:.4f}.pt"
                                
                                checkpoint_data = {
                                    "model": CoTModel.state_dict(),
                                    "global_step": global_step,
                                    "epoch": epoch_for_logging,
                                    "loss": train_loss,
                                }
                                
                                if not save_full_model:
                                    if 'optimizer' in globals():
                                        checkpoint_data["optimizer"] = optimizer.state_dict()
                                    if 'lr_scheduler' in globals():
                                        checkpoint_data["scheduler"] = lr_scheduler.state_dict()
                                
                                torch.save(checkpoint_data, checkpoint_path)
                                print(f"Checkpoint saved: {checkpoint_path}")
                                
                                wandb.log({"checkpoint/saved_step": global_step}, step=global_step)
                    
                    if world_size > 1:
                        dist.barrier() # Ensure all processes sync before next step if checkpoint was saved
                    print(f"global_step is: {global_step}; target_steps is: {target_steps}")
                    if global_step >= target_steps:
                        break # Break from inner (dataloader) loop

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

            # Explicitly log before checking the outer loop termination condition
            if overwatch.is_rank_zero():
                overwatch.info(f"DEBUG: End of epoch_idx {epoch_idx}. Current global_step = {global_step}, target_steps = {target_steps}")

            if global_step >= target_steps:
                if overwatch.is_rank_zero():
                    overwatch.info(f"Target steps reached after epoch {epoch_idx + 1}. Stopping training based on global_step ({global_step}) >= target_steps ({target_steps}).")
                break # Break from outer (epoch) loop
    
    if world_size > 1:
        cleanup_distributed()

   

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
new_special_tokens = ['<BOD>', '<EOD>']
cot_model = DiffusiveCoT(model, use_diff=True)
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

model.resize_token_embeddings(len(tokenizer))
dataset = GSM8KDataset("/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", tokenizer, 512, model, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusive CoT Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU') # Changed from default=1 to a more common value, adjust as needed
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller dataset')
    parser.add_argument('--data_path', type=str, default="/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", help='Path to training data JSON file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint file for resuming training')

    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and args.local_rank == -1:
        # If world_size is set (e.g., by torchrun) but local_rank is not passed via CLI,
        # try to get it from environment (torchrun sets LOCAL_RANK)
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if args.local_rank == -1:
            raise ValueError("torchrun is used (WORLD_SIZE > 1) but LOCAL_RANK is not set. Please ensure torchrun is configured correctly or pass --local_rank.")

    # It's crucial to set the start method for multiprocessing with CUDA correctly, especially for DDP.
    # 'spawn' is generally safer than 'fork' with CUDA.
    # This should be done once at the beginning of the script if __name__ == "__main__".
    if world_size > 1:
        try:
            mp.set_start_method('spawn', force=True)
            if world_size == 1 or args.local_rank == 0:
                print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            if world_size == 1 or args.local_rank == 0:
                print(f"Note: Could not set multiprocessing start method to 'spawn' (might be already set or not supported): {e}")

    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # model_for_dataset_init is only used for its tokenizer and device for embedding, not trained directly here.
    model_for_dataset_init = GPT2LMHeadModel.from_pretrained('openai-community/gpt2') 
    new_special_tokens = ['<BOD>', '<EOD>']
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    model_for_dataset_init.resize_token_embeddings(len(tokenizer))

    # The actual model to be trained
    gpt2_model_base = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    gpt2_model_base.resize_token_embeddings(len(tokenizer)) # Ensure this model also has resized embeddings
    cot_model = DiffusiveCoT(gpt2_model_base, use_diff=True)

    dataset = GSM8KDataset(args.data_path, tokenizer, 512, model_for_dataset_init, debug=args.debug)

    main(
        local_rank=args.local_rank,
        CoTModel=cot_model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        world_size=world_size,
        resume_from=args.resume_from
    )

