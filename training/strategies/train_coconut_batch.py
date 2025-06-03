import json
from torch.utils.data import Dataset
import torch




def run_vla_training(
    self,
    vla_dataset: IterableDataset,
    collator: PaddedCollatorForActionPrediction,
    metrics: VLAMetrics,
    save_interval: int = 2500,
    save_full_model: bool = True,
    use_diff: bool = False,
    repeated_diffusion_steps = 4,
    ar_diff_loss: bool = False,
) -> None:
    """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
    assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
    #assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

    # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
    dataloader = DataLoader(
        GSM8KDataset,
        batch_size=4,
        sampler=None,
        collate_fn=GSM8KDataset.collate_fn
    )

    # === Train ===
    status = metrics.get_status()
    sequence_length = 1024
    with tqdm(
        total=(self.epochs * math.ceil((len(dataloader) / overwatch.world_size() / self.grad_accumulation_steps))) if self.max_steps is None else self.max_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        self.vlm.train()

        self.optimizer.zero_grad()
        for train_idx, batch in enumerate(dataloader):
            # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
            #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
            predict_position = random.randint(0, sequence_length - 1)
            predict_mask = torch.ones_like(batch["input_ids"].shape)
            predict_mask[:, predict_position:] = False
            with torch.autocast(
                "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
            ):
                next_thought = self.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states = True,
                    repeated_diffusion_steps = repeated_diffusion_steps,
                    use_diff=True
                )

            # Commit Loss =>> Backward!
        
            metrics.commit(loss=loss)
            
            normalized_loss = loss / self.grad_accumulation_steps
            normalized_loss.backward()

            # === Gradient Step ===
            # Step =>> Only if Done w/ Gradient Accumulation
            if (train_idx + 1) % self.grad_accumulation_steps == 0:
                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                # Compute epoch value using number of completed gradient steps
                div = (len(vla_dataset) // self.global_batch_size) if (len(vla_dataset) // self.global_batch_size)!=0 else 1
                epoch = (metrics.global_step + 1) // div

                # Push Metrics
                metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                target_steps = self.epochs * math.ceil(len(dataloader) / overwatch.world_size() / self.grad_accumulation_steps)
                # Check if global_step is a fraction (1/10, 2/10, ..., 10/10) of target_steps
                for i in range(1, 6):
                    if metrics.global_step == target_steps * i / 5:
                        self.save_checkpoint(
                            metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                        )
                        dist.barrier()

                if metrics.global_step>=(self.epochs * math.ceil((len(dataloader) / overwatch.world_size() / self.grad_accumulation_steps))):
                    return

            # Update Progress Bar
            progress.update()
            progress.set_description(status)





class GSM8KDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=512):
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

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回一个tokenized后的样本"""
        sample = self.data[idx]
        
        # 提取原始文本字段
        question = sample['question']
        answer = sample['answer']
        steps = sample['steps']
        
        # 将所有步骤拼接成一个字符串
        steps_str = " ".join(steps)
        
        # 如果没有提供tokenizer，则返回原始文本
        if self.tokenizer is None:
            return {
                'question': question,
                'answer': answer,
                'steps': steps_str
            }
        
        # 使用tokenizer处理问题和答案+步骤的组合
        inputs = self.tokenizer(
            text=question,
            text_pair=f"{answer} [SEP] {steps_str}",  # 将答案和步骤合并
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 返回tokenized后的结果
        return {
            'input_ids': inputs['input_ids'].squeeze(0),       # 形状: (max_length,)
            'attention_mask': inputs['attention_mask'].squeeze(0), # 形状: (max_length,)
            'question': question,       # 保留原始文本用于可视化
            'answer': answer,
            'steps': steps
        }

    def collate_fn(self, batch):
        """
        自定义批处理函数（可选，可在DataLoader中使用）
        将多个样本堆叠成批次张量
        """
        if self.tokenizer is None:
            return batch
            
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        steps_list = [item['steps'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'questions': questions,
            'answers': answers,
            'steps_list': steps_list
        }