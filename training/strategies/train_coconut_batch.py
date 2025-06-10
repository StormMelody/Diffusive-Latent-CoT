import json
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
    answers = torch.stack(answers, dim=0)
    
    # Handle variable-length steps using padding
    # Pad steps to the same length
    steps_padded = rnn_utils.pad_sequence(steps_list, batch_first=True, padding_value=0)
    
    return questions, answers, steps_padded
class GSM8KDataset(Dataset):

    def __init__(self, data, tokenizer=None, max_length=512, embedding_model=None):
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
        # 获取embedding_model的设备
        device = next(self.embedding_model.parameters()).device
        # self.tokenizer(question, padding=True, truncation=True, return_tensors='pt') 方法会返回一个字典，包含了分词后的各种信息（如 input_ids 、 attention_mask 等）。
        # ** 则会将这个字典解包，并作为关键字参数传递给 embedding_model 方法，以获取对应的嵌入向量。
        # 最后进行池化，就变成了[1, 1, 768]
        question = self.embedding_model(**self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state'].mean(dim=1)
        answer = self.embedding_model(**self.tokenizer(answer, padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state'].mean(dim=1)
        # 使用output_hidden_states=True获取隐藏状态
        # question_output = self.embedding_model(**self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
        # question = question_output.hidden_states[-1].mean(dim=1)  # 使用最后一层的hidden_states
        # answer_output = self.embedding_model(**self.tokenizer(answer, padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
        # answer = answer_output.hidden_states[-1].mean(dim=1)  # 使用最后一层的hidden_states
        steps = []  # 用于存储处理后的steps
        for i in range(len(original_steps)):  # 使用original_steps的长度
            # 去掉.mean(dim=1)，直接使用.squeeze()来移除多余维度
            step_embedding = self.embedding_model(**self.tokenizer(original_steps[i], padding=True, truncation=True, return_tensors='pt').to(device))['last_hidden_state']
            # step_output = self.embedding_model(**self.tokenizer(original_steps[i], padding=True, truncation=True, return_tensors='pt').to(device), output_hidden_states=True)
            # step_embedding = step_output.hidden_states[-1]  # 使用最后一层的hidden_states
            # 取第一个token的embedding或者mean pooling，但保持2维
            # 取第一个token的embedding或者mean pooling，但保持2维
            steps.append(step_embedding.squeeze(0).mean(dim=0))  # [768]
        steps = torch.stack(steps, dim=0)  # [num_steps, 768]
        # 返回tokenized后的结果
        return question, answer, steps

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def main(
    CoTModel: DiffusiveCoT,
    dataset: GSM8KDataset,
    save_interval: int = 2500,
    save_full_model: bool = True,
    epochs: int = 10,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    max_steps: Optional[int] = None,
    enable_mixed_precision_training: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16
    ) -> None:

    dataloader = DataLoader(
        dataset,
        batch_size=16,
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
    num_training_steps = (len(dataset) * epochs) // (4 * grad_accumulation_steps)  # 4是batch_size
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
                "batch_size": 16,
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
        for train_idx, batch in enumerate(dataloader):
            # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
            #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
            question, answer, steps = batch     # shape: [4, 1, 768] [4, 1, 768] [4, 4, 768]
            device = next(CoTModel.parameters()).device
            question = question.to(device)
            answer = answer.to(device)
            steps = steps.to(device)
            with torch.autocast(
                "cuda", dtype=mixed_precision_dtype, enabled=enable_mixed_precision_training
            ):
                loss = CoTModel(
                    question=question,
                    steps=steps,
                    answer=answer,
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
                # Compute epoch value using number of completed gradient steps
                # div = (len(vla_dataset) // global_batch_size) if (len(vla_dataset) // global_batch_size)!=0 else 1
                # epoch = (metrics.global_step + 1) // div
                global_step += 1
                epoch = global_step // (len(dataloader) // grad_accumulation_steps) if (len(dataloader) // grad_accumulation_steps) > 0 else 0
                # 直接使用wandb记录
                if overwatch.is_rank_zero():
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }, step=global_step)
                    
                    # 打印训练信息
                    print(f"Step {global_step}/{target_steps}, Epoch {epoch}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                # Check if global_step is a fraction (1/10, 2/10, ..., 10/10) of target_steps
                # for i in range(1, 6):
                #     if metrics.global_step == target_steps * i / 5:
                #         save_checkpoint(
                #             metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                #         )
                #         dist.barrier()
                # 保存checkpoint
                for i in range(1, 6):
                    if global_step == target_steps * i // 5:
                        if overwatch.is_rank_zero():
                            checkpoint_dir = Path("./checkpoints")
                            checkpoint_dir.mkdir(exist_ok=True)
                            
                            train_loss = loss.item()
                            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                            
                            checkpoint_data = {
                                "model": CoTModel.state_dict(),
                                "global_step": global_step,
                                "epoch": epoch,
                                "loss": train_loss,
                            }
                            
                            if not save_full_model:
                                if 'optimizer' in globals():
                                    checkpoint_data["optimizer"] = optimizer.state_dict()
                                if 'lr_scheduler' in globals():
                                    checkpoint_data["scheduler"] = lr_scheduler.state_dict()
                            
                            torch.save(checkpoint_data, checkpoint_path)
                            print(f"Checkpoint saved: {checkpoint_path}")
                            
                            # 记录checkpoint到wandb
                            wandb.log({"checkpoint/saved_step": global_step}, step=global_step)
                        
                        if 'dist' in globals():
                            dist.barrier()
                
                if global_step >= target_steps:
                    break

            # Update Progress Bar
            progress.update()
            progress.set_description(status)

   

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('openai-community/gpt2')
new_special_tokens = ['<BOD>', '<EOD>']
cot_model = DiffusiveCoT(model, use_diff=True)
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

model.resize_token_embeddings(len(tokenizer))
dataset = GSM8KDataset("/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", tokenizer, 512, model)

main(cot_model, dataset, save_interval=2500, save_full_model=True)

