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
import pickle

def custom_collate_fn_preprocessed(batch):
    """
    自定义collate函数，用于处理预处理的embedding数据
    """
    questions, answers, steps_list = zip(*batch)
    
    # Stack questions (预处理的embeddings)
    questions = torch.stack(questions, dim=0).squeeze(1)  # [batch_size, 768]
    
    # 处理answers (token ids)
    answers = [tensor.squeeze(0) for tensor in answers]
    answers_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=50256)
    answers_label_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=-100)
    
    # 处理steps (预处理的embeddings)
    steps_padded = rnn_utils.pad_sequence(steps_list, batch_first=True, padding_value=0)
    
    return questions, answers_padded, steps_padded, answers_label_padded

class PreprocessedGSM8KDataset(Dataset):
    """
    使用预处理embedding的GSM8K数据集类
    避免在训练时重复计算embeddings，解决GPU访问冲突和性能问题
    """
    
    def __init__(self, preprocessed_data_path, debug=False, debug_samples=4):
        """
        初始化数据集
        :param preprocessed_data_path: 预处理后的pickle文件路径
        :param debug: 是否启用调试模式
        :param debug_samples: 调试模式下的样本数量
        """
        print(f"Loading preprocessed data from {preprocessed_data_path}...")
        with open(preprocessed_data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        if debug:
            self.data = self.data[:debug_samples]
            print(f"Debug mode: using {len(self.data)} samples")
        
        print(f"Loaded {len(self.data)} preprocessed samples")
    
    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本，直接返回预处理的embeddings
        不再进行任何GPU计算，避免多进程冲突
        """
        sample = self.data[idx]
        
        # 直接返回预处理的数据
        question_embedding = sample['question_embedding']  # [1, 768]
        answer_token_ids = sample['answer_token_ids']      # tokenized answer
        steps_embeddings = sample['steps_embeddings']     # [num_steps, 768]
        
        return question_embedding, answer_token_ids, steps_embeddings
    
    def get_original_data(self, idx):
        """
        获取原始文本数据，用于调试和验证
        """
        sample = self.data[idx]
        return {
            'question': sample['original_question'],
            'answer': sample['original_answer'],
            'steps': sample['original_steps']
        }

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = None # Will be initialized in main() with distributed parameters

def setup_distributed(rank, world_size):
    """Sets up the distributed training environment."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleans up the distributed training environment."""
    dist.destroy_process_group()

def train_with_preprocessed_embeddings(
    local_rank: int,
    CoTModel: DiffusiveCoT,
    preprocessed_data_path: str,
    save_interval: int = 2500,
    save_full_model: bool = True,
    epochs: int = 1,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    max_steps: Optional[int] = None,
    enable_mixed_precision_training: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    world_size: int = 1,
    debug: bool = False,
    debug_samples: int = 4
) -> None:
    """
    使用预处理embedding进行训练的主函数
    
    Args:
        local_rank: 本地GPU rank
        CoTModel: DiffusiveCoT模型
        preprocessed_data_path: 预处理数据文件路径
        save_interval: 保存间隔
        save_full_model: 是否保存完整模型
        epochs: 训练轮数
        grad_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        batch_size: 批次大小
        max_steps: 最大训练步数
        enable_mixed_precision_training: 是否启用混合精度训练
        mixed_precision_dtype: 混合精度数据类型
        world_size: 总GPU数量
        debug: 是否启用调试模式
        debug_samples: 调试模式样本数
    """
    
    if world_size > 1:
        setup_distributed(local_rank, world_size)

    global overwatch
    overwatch = initialize_overwatch("train_coconut_preprocessed")
    saved_steps = set()
    
    # 初始化训练日志记录
    training_logs = []
    
    # 初始化实时日志文件
    if overwatch.is_rank_zero():
        log_dir = Path("./training_logs")
        log_dir.mkdir(exist_ok=True)
        realtime_log_path = log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 写入日志文件头部信息
        with open(realtime_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Training started at: {datetime.now().isoformat()}\n")
            f.write(f"Configuration: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}\n")
            f.write(f"Preprocessed data: {preprocessed_data_path}\n")
            f.write("=" * 80 + "\n")
            f.write("Step\tEpoch\tLoss\t\tLearning_Rate\tTimestamp\n")
            f.write("-" * 80 + "\n")
        
        print(f"Real-time training log will be saved to: {realtime_log_path}") 
    # 创建使用预处理embedding的数据集
    dataset = PreprocessedGSM8KDataset(
        preprocessed_data_path=preprocessed_data_path,
        debug=debug,
        debug_samples=debug_samples
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
    
    # 使用预处理数据的DataLoader，可以安全地使用多进程
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn_preprocessed,  # 使用专门的collate函数
        shuffle=False if sampler else True,
        num_workers=4,  # 可以安全地使用多进程，因为不再有GPU计算
        pin_memory=True
    )
    
    import math
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    CoTModel = CoTModel.to(device)
    
    if world_size > 1:
        CoTModel = DDP(CoTModel, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = AdamW(CoTModel.parameters(), lr=learning_rate)
    print(f"Learning rate in optimizer: {optimizer.param_groups[0]['lr']}")
    
    # 计算训练步数
    if world_size > 1:
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    else:
        target_steps_per_epoch = math.ceil(len(dataloader) / grad_accumulation_steps)
    target_steps = epochs * target_steps_per_epoch
    
    num_training_steps = target_steps
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.99
    )
    
    # 初始化wandb
    if overwatch.is_rank_zero():
        wandb.init(
            mode="offline",
            project="diffusive-cot-preprocessed",
            name=f"train-preprocessed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "batch_size": batch_size,
                "grad_accumulation_steps": grad_accumulation_steps,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "preprocessed_data": True,
                "debug": debug
            }
        )
    
    # === Train ===
    status = "Training DiffusiveCoT with Preprocessed Embeddings"
    global_step = 0
    
    with tqdm(
        total=(epochs * math.ceil((len(dataloader) / overwatch.world_size() / grad_accumulation_steps))) if max_steps is None else max_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        CoTModel.train()
        optimizer.zero_grad()
        
        for epoch_idx in range(epochs):
            if overwatch.is_rank_zero():
                overwatch.info(f"Starting Epoch {epoch_idx + 1}/{epochs}")
            
            if world_size > 1 and hasattr(dataloader.sampler, "set_epoch") and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch_idx)

            for train_idx, batch in enumerate(dataloader):
                if overwatch.is_rank_zero():
                    print(f"train_idx is: {train_idx}")
                
                question, answer, steps, answer_label = batch
                
                # 移动数据到设备
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
                        output_hidden_states=True,
                    )

                normalized_loss = loss / grad_accumulation_steps
                normalized_loss.backward()

                # === Gradient Step ===
                if (train_idx + 1) % grad_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    epoch_for_logging = global_step // (len(dataloader) // grad_accumulation_steps) if (len(dataloader) // grad_accumulation_steps) > 0 else 0
                    
                    # 记录训练指标
                    if overwatch.is_rank_zero():
                        current_lr = lr_scheduler.get_last_lr()[0]
                        current_loss = loss.item()
                        
                        # 记录到训练日志
                        current_time = datetime.now()
                        log_entry = {
                            "global_step": global_step,
                            "epoch": epoch_idx + 1,
                            "log_epoch": epoch_for_logging,
                            "loss": current_loss,
                            "learning_rate": current_lr,
                            "timestamp": current_time.isoformat()
                        }
                        training_logs.append(log_entry)
                        
                        # 实时写入日志文件
                        with open(realtime_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{global_step}\t{epoch_idx + 1}\t{current_loss:.4f}\t\t{current_lr:.6f}\t\t{current_time.strftime('%H:%M:%S')}\n")
                            f.flush()  # 确保立即写入磁盘
                        
                        wandb.log({
                            "train/loss": current_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch_for_logging,
                            "train/global_step": global_step,
                        }, step=global_step)
                        
                        print(f"Step {global_step}/{target_steps}, Epoch {epoch_idx + 1}/{epochs} (LogEpoch: {epoch_for_logging}), Loss: {current_loss:.4f}, LR: {current_lr:.6f}")
                    
                    # 保存checkpoint
                    if overwatch.is_rank_zero():
                        for i in range(1, 6):  # Save 5 times during training
                            save_step = target_steps * i // 5
                            if global_step == save_step and save_step not in saved_steps:
                                saved_steps.add(save_step)
                                if global_step == target_steps * i // 5:
                                    checkpoint_dir = Path("./checkpoints_preprocessed")
                                    checkpoint_dir.mkdir(exist_ok=True)
                                    
                                    train_loss = loss.item()
                                    checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch_for_logging:02d}-loss={train_loss:.4f}.pt"
                                    
                                    checkpoint_data = {
                                        "model": CoTModel.state_dict(),
                                        "global_step": global_step,
                                        "epoch": epoch_for_logging,
                                        "loss": train_loss,
                                    }
                                    
                                    if not save_full_model:
                                        checkpoint_data["optimizer"] = optimizer.state_dict()
                                        checkpoint_data["scheduler"] = lr_scheduler.state_dict()
                                    
                                    torch.save(checkpoint_data, checkpoint_path)
                                    print(f"Checkpoint saved: {checkpoint_path}")
                                    
                                    wandb.log({"checkpoint/saved_step": global_step}, step=global_step)
                    
                    if world_size > 1:
                        dist.barrier()
                    
                    if overwatch.is_rank_zero():
                        print(f"global_step is: {global_step}; target_steps is: {target_steps}")
                    
                    if global_step >= target_steps:
                        break

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

            if overwatch.is_rank_zero():
                overwatch.info(f"DEBUG: End of epoch_idx {epoch_idx}. Current global_step = {global_step}, target_steps = {target_steps}")

            if global_step >= target_steps:
                if overwatch.is_rank_zero():
                    overwatch.info(f"Target steps reached after epoch {epoch_idx + 1}. Stopping training based on global_step ({global_step}) >= target_steps ({target_steps}).")
                break
    
    # 保存训练日志为JSON文件
    if overwatch.is_rank_zero() and training_logs:
        log_dir = Path("./training_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_path = log_dir / log_filename
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "preprocessed_data_path": preprocessed_data_path,
                    "debug": debug
                },
                "training_logs": training_logs,
                "total_steps": len(training_logs),
                "final_loss": training_logs[-1]["loss"] if training_logs else None,
                "final_lr": training_logs[-1]["learning_rate"] if training_logs else None
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Training logs saved to: {log_path}")
        overwatch.info(f"Training completed. Logs saved to {log_path}")
    
    if world_size > 1:
        cleanup_distributed()

def main():
    """
    主函数，用于演示如何使用预处理embedding进行训练
    """
    parser = argparse.ArgumentParser(description='Diffusive CoT Training with Preprocessed Embeddings')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs for distributed training')
    parser.add_argument('--preprocessed_data_path', type=str, required=True, help='Path to preprocessed embeddings pickle file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug_samples', type=int, default=16, help='Number of samples in debug mode')
    
    args = parser.parse_args()
    
    # 初始化tokenizer和模型
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    
    # 添加特殊token
    new_special_tokens = ['<BOD>', '<EOD>']
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # 创建DiffusiveCoT模型
    cot_model = DiffusiveCoT(model, use_diff=True)
    
    if args.world_size > 1:
        mp.spawn(
            train_with_preprocessed_embeddings,
            args=(
                cot_model,
                args.preprocessed_data_path,
                2500,  # save_interval
                True,  # save_full_model
                args.epochs,
                args.grad_accumulation_steps,
                args.learning_rate,
                args.batch_size,
                None,  # max_steps
                True,  # enable_mixed_precision_training
                torch.bfloat16,  # mixed_precision_dtype
                args.world_size,
                args.debug,
                args.debug_samples
            ),
            nprocs=args.world_size,
            join=True
        )
    else:
        train_with_preprocessed_embeddings(
            local_rank=0,
            CoTModel=cot_model,
            preprocessed_data_path=args.preprocessed_data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            grad_accumulation_steps=args.grad_accumulation_steps,
            debug=args.debug,
            debug_samples=args.debug_samples
        )

if __name__ == "__main__":
    main()