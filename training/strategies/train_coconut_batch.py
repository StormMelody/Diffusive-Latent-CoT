import json
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

def custom_collate_fn(batch):
    questions, answers, steps_list = zip(*batch)
    
    # Stack questions and answers (these should have consistent shapes)
    questions = torch.stack(questions, dim=0)
    answers = [tensor.squeeze(0) for tensor in answers]
    answers_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=50256)
    ansers_label_padded = rnn_utils.pad_sequence(answers, batch_first=True, padding_value=-100)
    # Handle variable-length steps using padding
    # Pad steps to the same length
    steps_padded = rnn_utils.pad_sequence(steps_list, batch_first=True, padding_value=0)
    
    return questions, answers_padded, steps_padded, ansers_label_padded
class GSM8KDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_length=512, embedding_model=None,debug=False, debug_samples=4):
        """
        初始化数据集
        :param data: 可以是包含样本的列表，或JSON文件的路径
        :param tokenizer: 用于文本编码的tokenizer（如Hugging Face的tokenizer）
        :param max_length: 输入序列的最大长度
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

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 提取原始文本字段
        '''
        sample = {'question': 'Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?', 'steps': ['<<600*30/100=180>>', '<<600*10/100=60>>', '<<180+60=240>>', '<<600-240=360>>'], 'answer': '360'}
        question = 'Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?'
        '''
        question = sample['question']
        answer = sample['answer']
        original_steps = []
        original_steps.append('<BOD>')
        original_steps.extend(sample['steps'])  # 保存原始steps数据
        original_steps.append('<EOD>')
        # 检查steps是否为空
        if len(sample['steps']) == 0:
            print(f"Warning: Empty steps found for item {idx}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            # 递归调用下一个索引，直到找到有效的样本或达到数据集末尾
            if idx + 1 < len(self.data):
                return self.__getitem__(idx + 1)
            else:
                # 如果已经是最后一个样本，返回一个默认值或第一个样本
                return self.__getitem__(0)
                
        # 获取embedding_model的设备
        device = next(self.embedding_model.parameters()).device
        # self.tokenizer(question, padding=True, truncation=True, return_tensors='pt') 方法会返回一个字典，包含了分词后的各种信息（如 input_ids 、 attention_mask 等）。
        # ** 则会将这个字典解包，并作为关键字参数传递给 embedding_model 方法，以获取对应的嵌入向量。
        # 最后进行池化，就变成了[1, 1, 768]
        # question = self.embedding_model(**self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state'].mean(dim=1)
        # answer = self.embedding_model(**self.tokenizer(answer, padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state'].mean(dim=1)
        # 使用output_hidden_states=True获取隐藏状态
        question_output = self.embedding_model(**self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
        question = question_output.hidden_states[-1].mean(dim=1)  # 使用最后一层的hidden_states
        # answer_output = self.embedding_model(**self.tokenizer(answer, padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
        # answer = answer_output.hidden_states[-1].mean(dim=1)  # 使用最后一层的hidden_states
        tokenizer_output = tokenizer(answer + tokenizer.eos_token, padding=True, truncation=True, return_tensors='pt')
        answer_token_ids = tokenizer_output.input_ids
        steps = []  # 用于存储处理后的steps
        for i in range(len(original_steps)):  # 使用original_steps的长度
            # 检查步骤是否为空字符串，如果是则跳过
            if not original_steps[i]:
                print(f"Warning: Empty step found at position {i} for item {idx}")
                continue
                
            try:
                # 去掉.mean(dim=1)，直接使用.squeeze()来移除多余维度
                # step_embedding = self.embedding_model(**self.tokenizer(original_steps[i], padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state']
                step_output = self.embedding_model(**self.tokenizer(original_steps[i], padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
                step_embedding = step_output.hidden_states[-1]  # 使用最后一层的hidden_states
                # 取第一个token的embedding或者mean pooling，但保持2维
                # 取第一个token的embedding或者mean pooling，但保持2维
                steps.append(step_embedding.squeeze(0).mean(dim=0))  # [768]
            except Exception as e:
                print(f"Error processing step {i} for item {idx}: {e}")
                print(f"Step content: '{original_steps[i]}'")
                # 如果处理步骤时出错，跳过该步骤
                continue
                
        # 确保至少有一个步骤，否则递归调用下一个样本
        if len(steps) == 0:
            print(f"Warning: No valid steps found for item {idx}")
            if idx + 1 < len(self.data):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
                
        steps = torch.stack(steps, dim=0)  # [num_steps, 768]
        # 返回tokenized后的结果
        return question, answer_token_ids, steps

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def main(
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
    mixed_precision_dtype: torch.dtype = torch.bfloat16
    ) -> None:

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        collate_fn=custom_collate_fn
        # collate_fn=GSM8KDataset.collate_fn
    )
    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import get_cosine_schedule_with_warmup  # 添加这一行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CoTModel=CoTModel.to(device)
    optimizer = AdamW(CoTModel.parameters(), lr=1e-4)
    print(f"Learning rate in optimizer: {optimizer.param_groups[0]['lr']}")
    # 计算训练步数
    num_training_steps = (len(dataset) * epochs) // (batch_size * grad_accumulation_steps)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    # 设置warmup步数，通常为总训练步数的10%
    num_warmup_steps = int(0.1 * num_training_steps)
    
    # 使用带有warmup的余弦学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
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
    sequence_length = 1024
    target_steps = epochs * math.ceil(len(dataloader) / overwatch.world_size() / grad_accumulation_steps)
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
            
            # If using DistributedSampler, set epoch for dataloader sampler:
            # if hasattr(dataloader.sampler, "set_epoch") and isinstance(dataloader.sampler, DistributedSampler):
            #     dataloader.sampler.set_epoch(epoch_idx)

            for train_idx, batch in enumerate(dataloader):
                print("train_idx is: ", train_idx)
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                question, answer, steps, answer_label = batch     # shape: [4, 1, 768] [4, 1, 768] [4, 4, 768]
                # import pdb; pdb.set_trace()
                device = next(CoTModel.parameters()).device
                question = question.to(device)
                answer = answer.to(device)
                steps = steps.to(device)
                answer_label = answer_label.to(device)
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
                    for i in range(1, 6):
                        if global_step == target_steps * i // 5:
                            if overwatch.is_rank_zero():
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
                            
                            if 'dist' in globals():
                                dist.barrier()
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

   

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
new_special_tokens = ['<BOD>', '<EOD>']
cot_model = DiffusiveCoT(model, use_diff=True)
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

model.resize_token_embeddings(len(tokenizer))
dataset = GSM8KDataset("/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", tokenizer, 512, model, debug=True)

main(cot_model, dataset, save_interval=2500, save_full_model=True, epochs=3)

