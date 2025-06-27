# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from DiffusiveCoT import DiffusiveCoT
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
os.environ["WANDB_MODE"] = "offline"

def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    # dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    # tokenizer 作用：当输入文本时，tokenizer会：将文本分割成token；将token转换为对应的数字ID；识别并处理特殊token（如我们新添加的这些标记）
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token   # 使用结束标记作为填充标记，当模型看到填充部分时，会将其视为序列结束的信号。在处理批量数据时，由于每个序列长度可能不同，需要将所有序列填充到相同长度。- 序列1: "今天天气真好" (5个token)序列2: "我喜欢" (3个token)- 为了批处理，需要将序列2填充到5个token：["我","喜欢", PAD, PAD, PAD]
    tokenizer.add_tokens("<BOD>")
    tokenizer.add_tokens("<EOD>")  
    start_id = tokenizer.convert_tokens_to_ids("<BOD>") # 将这个标识转化为id，这个id后续会添加到embedding查找表中。id可以通过索引寻找对应的embedding。
    end_id = tokenizer.convert_tokens_to_ids("<EOD>")

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

    # 修改embeddings层：当不使用CoT(Chain of Thought)模式、不禁用thoughts、不禁用CoT时才执行，即使用完整的Coconut模式
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))   # 调整模型的embedding层大小，因为我们之前添加了新的特殊token（ <|latent|> 等），所以需要扩展embedding层以容纳这些新token
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<") # 选择 "<<" 这个已有的 token 的 embedding 来初始化新添加的特殊 token 的 embedding。提供一个“中性”的起点： 选择 "<<" 这样一个标点符号或者特殊字符作为模板，可能是因为它相对来说没有特别强的语义含义。如果选择一个像“猫”或“很高兴”这样的词汇来初始化，新添加的特殊 token 就会继承这些词汇的语义信息，这可能会影响模型学习这些特殊 token 应该代表的真正功能（比如 <|latent|> 可能代表的是一种潜在空间的信息，而不是某个具体的词义）。
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [start_id, end_id]:
            target_embedding = embeddings.weight.data[token_id]
            embeddings.weight.data[token_id] = target_embedding # 简单地赋值，后续训练中会更新
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]  # 尝试更新语言模型头部（lm_head）中对应的权重


    if configs.coconut:
        model = DiffusiveCoT(model, tokenizer=tokenizer, use_diff=True)

    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, start_id, end_id, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, start_id, end_id, max_size=64 if configs.debug else 100000000
        )
    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    # Retrieve start_id and end_id which should be defined earlier in the script
    # based on the logic for adding special tokens to the tokenizer.
    # Assuming start_id and end_id are available in this scope.
    # If they are not, their definition/retrieval needs to be ensured.
    # For example, they might be defined similar to how latent_id is handled:
    # start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    # end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # collator = MyCollator(
    #     tokenizer, 
    #     # latent_id=latent_id, 
    #     start_id=start_id,  # Pass start_id
    #     end_id=end_id,      # Pass end_id
    #     label_pad_token_id=-100
    # )

    from functools import partial

    def custom_collate_fn(batch, tokenizer):
        input_ids_list = []
        labels_list = []

        for sample in batch:
            q_tok = sample['question_tokenized']
            s_tok_nested = sample['steps_tokenized']
            s_tok_flat = [token for step in s_tok_nested for token in step]
            a_tok = sample['answer_tokenized']

            input_ids = q_tok + s_tok_flat + a_tok
            labels = ([-100] * len(q_tok)) + s_tok_flat + a_tok

            input_ids_list.append(torch.tensor(input_ids))
            labels_list.append(torch.tensor(labels))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }

    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer)

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )   # 确定当前的训练阶段。Coconut使用课程学习策略，随着训练的进行，逐渐增加潜在思维标记的数量。
        # dataset_gen_val = get_question_latent_dataset(
        #     scheduled_stage,
        #     base_dataset_valid,
        #     configs,
        #     start_id,
        #     # latent_id,
        #     end_id,
        #     no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        # )

        # valid_gen_dataloader = torch.utils.data.DataLoader(
        #     dataset_gen_val,
        #     num_workers=1,
        #     pin_memory=True,
        #     batch_size=1,
        #     collate_fn=collator,
        #     sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        # )   # 准备用于生成评估的验证数据集，这个数据集只包含问题和潜在思维标记，用于测试模型的生成能力。

        if not configs.only_eval:
            # 准备训练数据集和用于计算验证损失的数据集。训练数据集包含问题、潜在思维标记、思维步骤和答案。
            # dataset_train = get_cot_latent_dataset(
            #     scheduled_stage,
            #     base_dataset_train,
            #     configs,
            #     start_id,
            #     # latent_id,
            #     end_id,
            #     no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            #     shuffle=True,
            # )

            train_dataloader = torch.utils.data.DataLoader(
                base_dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                sampler=DistributedSampler(base_dataset_train, shuffle=True),
                collate_fn=collate_fn,
            )
            # import pdb; pdb.set_trace()
            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            # dataset_loss_val = get_cot_latent_dataset(
            #     scheduled_stage,
            #     base_dataset_valid,
            #     configs,
            #     start_id,
            #     latent_id,
            #     end_id,
            #     no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            # )

            # valid_loss_dataloader = torch.utils.data.DataLoader(
            #     dataset_loss_val,
            #     num_workers=1,
            #     shuffle=False,
            #     pin_memory=True,
            #     batch_size=configs.batch_size_training,
            #     collate_fn=collator,
            #     sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            # )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()   # 将模型设置为训练模式

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                # batch.keys()=dict_keys(['idx', 'input_ids', 'attention_mask', 'labels', 'position_ids'])
                import pdb; pdb.set_trace()
                if step == 0 and wandb_run and rank == 0:   # 记录训练数据（仅在第一步）
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})
                # 前向传播和损失计算
                total_train_steps += 1
                # batch.keys()
                # dict_keys(['input_ids', 'attention_mask', 'question_tokenized', 'steps_tokenized', 'answer_tokenized', 'steps_ids_padded', 'answer_ids_padded', 'answer_labels_padded', 'labels', 'position_ids'])
                batch = {
                    key: batch[key].to(rank) if hasattr(batch[key], 'to') else batch[key] for key in batch.keys() if key != "idx"
                }
                # import pdb; pdb.set_trace()

                outputs = parallel_model(**batch)
                # outputs = parallel_model(question=batch['question_tokenized'], steps=batch['steps_tokenized'], answer=batch['answer_ids_padded'], answer_labels=batch['answer_labels_padded'])
                loss = outputs.loss / configs.gradient_accumulation_steps
                # import pdb; pdb.set_trace
                # loss = outputs / configs.gradient_accumulation_steps
                loss.backward()
                # 梯度累积和优化器步骤
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                # 记录训练指标
                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()
            # 模型保存
            # Always save at the end of each epoch if not in debug mode and not only_eval
            if (
                not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print(f"Saving model checkpoint at epoch {epoch + 1}.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

        #     # val loss
        #     total_loss = 0

        #     with torch.no_grad():
        #         parallel_model.module.eval()
        #         for step, batch in enumerate(valid_loss_dataloader):

        #             batch = {
        #                 key: batch[key].to(rank) if hasattr(batch[key], 'to') else batch[key] for key in batch.keys() if key != "idx"
        #             }

        #             outputs = parallel_model(**batch)
        #             loss = outputs.loss
        #             dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        #             total_loss += loss.item() / world_size

        #         if wandb_run and rank == 0:

        #             log_dict = {
        #                 "eval/loss": total_loss / len(valid_loss_dataloader),
        #             }
        #             wandb_run.log(log_dict)
        #             print("eval loss", total_loss / len(valid_loss_dataloader))

        # # val generation accuracy
        # total_length = len(valid_gen_dataloader)

        # pbar = tqdm(
        #     colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        # )
        # cor, cor_cot, total = (
        #     torch.tensor(0, device=rank),
        #     torch.tensor(0, device=rank),
        #     torch.tensor(0, device=rank),
        # )

        # with torch.no_grad():
        #     parallel_model.module.eval()
        #     for idx, batch in enumerate(valid_gen_dataloader):
        #         # 准备数据
        #         test_idx = batch["idx"][0]

        #         batch = {
        #             k: v.to(rank)
        #             for k, v in batch.items()
        #             if v != None and k not in ["idx", "position_ids"]
        #         }
        #         # https://github.com/huggingface/transformers/issues/32492
        #         # 获取参考答案
        #         assert len(batch["input_ids"]) == 1
        #         current_test_idx = test_idx.cpu().item() if torch.is_tensor(test_idx) else test_idx
        #         answer = answers_val[current_test_idx]
        #         answer_cot = cot_val[current_test_idx]
        #         question = question_val[current_test_idx]

        #         total += 1

        #         # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
        #         # 生成回答，现在分别是decoded_answers, generated_ids, generated_steps_representation
        #         outputs = parallel_model.module.generate(
        #             **batch,
        #             tokenizer=tokenizer, # Added tokenizer
        #             max_new_tokens=max_new_tokens,
        #             synced_gpus=not configs.only_eval,
        #         )
        #         import pdb; pdb.set_trace()
        #         # 解码和评估
        #         # text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #         # answer_output = text_output.split("#")[-1].replace(",", "").strip()
        #         # list转换为str
        #         print(question)
        #         print(outputs[0])
        #         answer_output = outputs[0][0] if isinstance(outputs[0], list) and outputs[0] else outputs[0]
        #         # cot_output = (
        #         #     ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
        #         # )

        #         # if idx < 5 and rank == 0:
        #         #     # print some examples
        #         #     print(
        #         #         f"Question {current_test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
        #         #     )
        #         #     print(f"Full output: '{tokenizer.decode(outputs[0])}'")
        #         #     print(f"Extracted Output: '{answer_output}'")
        #         # 计算准确率
        #         cor += answer_output == answer
        #         # cor_cot += cot_output == answer_cot

        #         pbar.update(1)
        #         pbar.set_description(
        #             f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
        #         )

        #     pbar.close()
        #     # print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")
        #     print(f"Device {rank}: Cor={cor}, Total={total}")
        # # 汇总多GPU上的评估结果，计算并记录最终的准确率指标。
        # # dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        # dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(total, op=dist.ReduceOp.SUM)

        # # cor_cot = cor_cot.item()
        # cor = cor.item()
        # total = total.item()
        # if rank == 0:
        #     print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
        #     # print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        # sys.stdout.flush()

        # if wandb_run:
        #     # wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})
        #     wandb_run.log({"eval/acc": cor / total})

        # if configs.only_eval:
        #     break

        # dist.barrier()
        # # 如果配置为仅保存改进的模型，且当前准确率超过了最佳准确率，则保存模型。
        # if (
        #     cor / total > best_acc
        #     and configs.save_only_improve
        #     and not configs.debug
        #     and not configs.only_eval
        # ):
        #     states = parallel_model.state_dict()

        #     if rank == 0:
        #         torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
        #         print("saving model.")

        #     best_acc = cor / total

        #     dist.barrier()
        #     del states
        #     gc.collect()
        #     torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
