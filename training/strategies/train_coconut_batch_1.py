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
import pickle # Added for saving/loading precomputed embeddings
import hashlib # Added for creating unique filenames for embeddings
import time # Added for profiling

class GSM8KDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_length=512, embedding_model=None, debug=False, debug_samples=32, embedding_store_dir="./embedding_cache", embedding_batch_size=32):
        """
        初始化数据集
        :param data: 可以是包含样本的列表，或JSON文件的路径
        :param tokenizer: 用于文本编码的tokenizer（如Hugging Face的tokenizer）
        :param max_length: 输入序列的最大长度
        :param embedding_model: 用于生成嵌入的模型
        :param debug: 是否使用调试模式（只使用少量样本）
        :param debug_samples: 调试模式下使用的样本数量
        :param embedding_batch_size: 构建嵌入缓存时批处理大小
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
        self.embedding_model = embedding_model # Still needed for initial cache generation
        self.device = next(embedding_model.parameters()).device if embedding_model is not None else 'cpu'
        self.embedding_store_dir = Path(embedding_store_dir)
        self.embedding_store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_batch_size = embedding_batch_size

        # 预处理和过滤数据，移除空steps的样本
        self._preprocess_data() # Ensure self.data is populated before generating/loading embeddings

        # Generate a unique filename based on the data content to avoid stale caches
        data_hash = hashlib.md5(json.dumps(self.data, sort_keys=True).encode('utf-8')).hexdigest()
        self.embedding_store_path = self.embedding_store_dir / f"embeddings_{data_hash}.pkl"

        if not self.embedding_store_path.exists() and (self.embedding_model is None or self.tokenizer is None):
            raise ValueError("Embedding model and tokenizer are required to build embedding store, but not provided, and no precomputed store found.")

        self._load_or_build_embedding_store()

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

    def _load_or_build_embedding_store(self):
        """Loads embeddings from a precomputed file or builds them if the file doesn't exist."""
        if self.embedding_store_path.exists():
            print(f"Loading precomputed embeddings from {self.embedding_store_path}")
            t_load_start = time.time()
            with open(self.embedding_store_path, 'rb') as f:
                self.embedding_store = pickle.load(f)
            t_load_end = time.time()
            print(f"Loaded {len(self.embedding_store)} embeddings in {t_load_end - t_load_start:.2f} seconds.")
        else:
            print(f"Precomputed embeddings not found at {self.embedding_store_path}. Building now...")
            overall_build_start_time = time.time()
            
            self.embedding_store = {}

            print(f"Using device: {self.device} for embedding generation.")

            collect_texts_start_time = time.time()
            unique_question_texts = set()
            unique_step_texts = set()  # Includes <BOD>, <EOD>, and actual steps

            for sample_item in self.data:
                unique_question_texts.add(sample_item['question'])
                for step_text_item in ['<BOD>'] + sample_item['steps'] + ['<EOD>']:
                    if step_text_item:  # Ensure step is not empty
                        unique_step_texts.add(step_text_item)
            
            texts_to_embed_as_questions = list(unique_question_texts)
            texts_to_embed_as_steps = list(unique_step_texts - unique_question_texts)
            collect_texts_end_time = time.time()
            print(f"Time to collect {len(texts_to_embed_as_questions)} unique question texts and {len(texts_to_embed_as_steps)} unique step texts: {collect_texts_end_time - collect_texts_start_time:.2f} seconds.")

            def _generate_embeddings_batch_inner(texts_list, is_step_for_pooling, pbar_desc, batch_size_override=None):
                if not texts_list:
                    return {}
                
                text_to_embedding_map = {}
                actual_batch_size = batch_size_override if batch_size_override is not None else self.embedding_batch_size
                num_batches = (len(texts_list) + actual_batch_size - 1) // actual_batch_size
                
                for i in tqdm(range(num_batches), desc=pbar_desc):
                    batch_texts = texts_list[i*actual_batch_size : (i+1)*actual_batch_size]
                    if not batch_texts: continue

                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs, output_hidden_states=True)
                        last_hidden_states = outputs.hidden_states[-1]

                    attention_mask = inputs.attention_mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_states).float()
                    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)
                    num_non_padded_tokens = attention_mask.sum(dim=1, keepdim=True)
                    num_non_padded_tokens = torch.clamp(num_non_padded_tokens, min=1e-9)
                    pooled_batch_gpu = sum_embeddings / num_non_padded_tokens
                    pooled_batch_cpu = pooled_batch_gpu.cpu()

                    for k_idx, text_content in enumerate(batch_texts):
                        embedding_cpu = pooled_batch_cpu[k_idx]
                        if is_step_for_pooling:
                            final_embedding = embedding_cpu.half()
                        else:
                            final_embedding = embedding_cpu.unsqueeze(0).half()
                        text_to_embedding_map[text_content] = final_embedding
                return text_to_embedding_map

            question_gen_start_time = time.time()
            # Calculate question_batch_size based on user's logic
            question_batch_size_for_call = self.embedding_batch_size
            if self.embedding_batch_size > 16: # Assuming this logic is desired
                question_batch_size_for_call = max(16, self.embedding_batch_size // 2)
            print(f"Using batch size {question_batch_size_for_call} for question embeddings (original: {self.embedding_batch_size}).") # Adapted print
            
            question_embeddings = _generate_embeddings_batch_inner(
                texts_to_embed_as_questions, 
                is_step_for_pooling=False, 
                pbar_desc="Building Question Embeddings",
                batch_size_override=question_batch_size_for_call # Pass the new parameter
            )
            self.embedding_store.update(question_embeddings)
            question_gen_end_time = time.time()
            print(f"Time to build question embeddings: {question_gen_end_time - question_gen_start_time:.2f} seconds.")

            step_gen_start_time = time.time()
            print(f"Using batch size {self.embedding_batch_size} for step embeddings.") # Adapted print
            
            step_embeddings = _generate_embeddings_batch_inner(
                texts_to_embed_as_steps, 
                is_step_for_pooling=True, 
                pbar_desc="Building Step Embeddings"
                # No batch_size_override is passed, so it will use the default (None)
                # and the function will use self.embedding_batch_size
            )
            self.embedding_store.update(step_embeddings)
            step_gen_end_time = time.time()
            print(f"Time to build step embeddings: {step_gen_end_time - step_gen_start_time:.2f} seconds.")
            
            save_start_time = time.time()
            with open(self.embedding_store_path, 'wb') as f:
                pickle.dump(self.embedding_store, f)
            save_end_time = time.time()
            print(f"Time to save embeddings to {self.embedding_store_path}: {save_end_time - save_start_time:.2f} seconds.")
            
            overall_build_end_time = time.time()
            print(f"Total time to build and save embeddings: {overall_build_end_time - overall_build_start_time:.2f} seconds.")
            print(f"Built and saved {len(self.embedding_store)} embeddings.")

        # Optionally, if embedding_model is large and only for precomputation and not needed later by this instance:
        if hasattr(self, 'embedding_model') and self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None # Ensure it's None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Released embedding model and cleared CUDA cache.")

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
        # 获取question的嵌入 from store
        question_embedding = self.embedding_store.get(question)
        if question_embedding is None:
            print(f"Warning: Embedding for question not found in store: {question}")
            # Fallback or error handling: try to generate on the fly (if model retained) or skip
            # For simplicity, we'll skip if not found after precomputation attempt.
            return self.__getitem__((idx + 1) % len(self.data)) # Or raise error
        
        # 处理answer的token ids
        tokenizer_output = self.tokenizer(answer + self.tokenizer.eos_token, padding=True, truncation=True, return_tensors='pt')
        answer_token_ids = tokenizer_output.input_ids
        
        # 处理steps的嵌入 from store
        steps_embeddings = []
        for step_text in original_steps:
            if not step_text:
                continue
            step_embedding = self.embedding_store.get(step_text)
            if step_embedding is not None:
                steps_embeddings.append(step_embedding)
            else:
                print(f"Warning: Embedding for step not found in store: {step_text}")
                # Optionally skip this step or the whole sample
        
        # 确保至少有一个步骤
        if len(steps_embeddings) == 0:
            print(f"Warning: No valid step embeddings found for item {idx} after checking store.")
            return self.__getitem__((idx + 1) % len(self.data))
        
        steps_tensor = torch.stack(steps_embeddings, dim=0)  # [num_steps, 768]
        
        # 返回处理后的结果 (ensure tensors are on the correct device if needed by collate_fn/model)
        # Embeddings are stored on CPU as float16. Convert to float32 for model consumption.
        # Collate_fn or training loop should move to GPU.
        if question_embedding is not None:
            question_embedding = question_embedding.float() # Convert to float32
        if steps_tensor is not None:
            steps_tensor = steps_tensor.float() # Convert to float32

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
    epochs: int = 1,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    batch_size=1,
    max_steps: Optional[int] = None,
    enable_mixed_precision_training: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    world_size: int = 1, # Added for DDP
    resume_from_checkpoint: Optional[str] = None # Added for resuming training
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
        num_workers=32, # Adjusted: Increase num_workers for parallel data loading
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
    print(f"Learning rate in optimizer: {optimizer.param_groups[0]['lr']}")
    # 计算训练步数
    num_training_steps = (len(dataset) * epochs) // (batch_size * grad_accumulation_steps)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    # 设置warmup步数，通常为总训练步数的10%
    num_warmup_steps = int(0.1 * num_training_steps)
    
    # 使用带有warmup的余弦学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.99
    )
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
                "learning_rate": learning_rate
            }
        )
    # === Train ===
    status = "Training DiffusiveCoT"
    global_step = 0
    start_epoch = 0 # Added for resuming training
    sequence_length = 1024

    # Load checkpoint if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if overwatch.is_rank_zero():
            overwatch.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        # Handle DDP model loading carefully
        model_to_load = CoTModel.module if world_size > 1 else CoTModel
        
        # Remove embedding.weight from checkpoint if it exists
        if 'embedding.weight' in checkpoint['model']:
            del checkpoint['model']['embedding.weight']
            if overwatch.is_rank_zero():
                overwatch.info("Removed 'embedding.weight' from checkpoint state_dict before loading.")

        model_to_load.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch'] # Ensure 'epoch' in checkpoint is the completed epoch number
        if overwatch.is_rank_zero():
            overwatch.info(f"Resumed from global_step: {global_step}, start_epoch: {start_epoch + 1}")
    elif resume_from_checkpoint:
        if overwatch.is_rank_zero():
            overwatch.warning(f"Checkpoint {resume_from_checkpoint} not found. Starting from scratch.")

    # Adjust num_training_steps and target_steps for DDP
    # Each process sees a fraction of the data, but grad_accumulation_steps applies per process
    # Total effective batch size = batch_size * world_size * grad_accumulation_steps
    # num_training_steps_per_epoch = len(dataset) // (batch_size * world_size * grad_accumulation_steps)
    # num_training_steps = num_training_steps_per_epoch * epochs
    # However, len(dataloader) already considers the sharding by DistributedSampler if used.
    # So, len(dataloader) is len(dataset) / world_size for DDP.
    # The original calculation for num_training_steps seems correct if len(dataloader) is used.
    # Let's re-verify target_steps calculation for tqdm
    if world_size > 1:
        # For DDP, len(dataloader) is len(full_dataset) / world_size
        # So, total batches processed by all gpus in one epoch is len(dataloader) * world_size
        # Number of optimizer steps per epoch = (len(dataloader) * world_size) / (world_size * grad_accumulation_steps)
        # = len(dataloader) / grad_accumulation_steps
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    else:
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    target_steps = epochs * target_steps_per_epoch

    # num_training_steps should be calculated based on the total number of optimizer steps
    num_training_steps = target_steps # This seems more direct

    # Re-initialize lr_scheduler with the potentially updated num_training_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    with tqdm(
        total=(epochs * math.ceil((len(dataloader) / overwatch.world_size() / grad_accumulation_steps))) if max_steps is None else max_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        CoTModel.train()
        optimizer.zero_grad() # Moved here to ensure it's called after potential optimizer.load_state_dict
        for epoch_idx in range(start_epoch, epochs): # Start from start_epoch
            if overwatch.is_rank_zero():
                overwatch.info(f"Starting Epoch {epoch_idx + 1}/{epochs}")
            
            if world_size > 1 and hasattr(dataloader.sampler, "set_epoch") and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch_idx)

            for train_idx, batch in enumerate(dataloader):
                print("train_idx is: ", train_idx)
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                # question 是问题的embedding区平均值，[4,768]
                # answer 还是token的形式，然后一个an
                # steps 都是对齐之后加上0
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
                                    "model": CoTModel.module.state_dict() if world_size > 1 else CoTModel.state_dict(), # Save module for DDP
                                    "optimizer": optimizer.state_dict(), # Always save optimizer and scheduler for resuming
                                    "scheduler": lr_scheduler.state_dict(),
                                    "global_step": global_step,
                                    "epoch": epoch_idx, # Save current epoch_idx (0-indexed)
                                    "loss": train_loss,
                                }
                                
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
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')

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
        resume_from_checkpoint=args.resume_from_checkpoint # Pass resume_from_checkpoint
    )

