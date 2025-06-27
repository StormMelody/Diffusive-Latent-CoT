# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, start_id, end_id, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        steps_tokenized = [tokenizer.encode("<BOD>", add_special_tokens=False)] + [tokenizer.encode("\n", add_special_tokens=False)]+ steps_tokenized + [tokenizer.encode("<EOD>", add_special_tokens=False)] + [tokenizer.encode("\n", add_special_tokens=False)]    # <BOE>后面加上换行符号
        # 结构为
        # <Question> \n
        # <BOD> \n
        # <Step1> \n
        # <Step2> \n
        # ...
        # <EOD> \n
        # ### <Answer> <EOS>
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # .map() 方法将 tokenize_sample 函数应用到数据集中的每个样本上。在这个上下文中，它将原始文本数据（问题、步骤和答案）转换为模型可以处理的标记化（tokenized）形式。
            # remove_columns=list(dataset.features) 参数表示在处理后移除原始数据集中的所有列，只保留 tokenize_sample 函数返回的新列
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )
    # verify,通过第0个元素来判断得到的dataset[0]是否和complete的范式一样
    d = data[0]
    # "\n".join(...) 是一个 Python 字符串方法，它会将列表中的所有字符串元素用换行符 ( \n ) 连接成一个单一的字符串
    complete = d["question"] + "\n" + "<BOD>" + "\n" + "\n".join(d["steps"]) + "\n" + "<EOD>"+ "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    steps_list = []

    for i in range(len(dataset[0]["steps_tokenized"])):
        steps_list += dataset[0]["steps_tokenized"][i]
    dataset_0_list = dataset[0]["question_tokenized"] + steps_list + dataset[0]["answer_tokenized"]
    # import pdb; pdb.set_trace()
    assert (complete_tokenized == dataset_0_list)
    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    start_id: Optional[int] = None
    latent_id: Optional[int] = None
    end_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):

    def process_dataset(sample):

        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)

        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage,    # 当前训练阶段，控制使用多少潜在思维标记
    base_dataset,       # 基础数据集，包含已分词的问题、步骤和答案
    configs,
    start_id,           # 开始潜在思维的特殊标记ID <|start-latent|>, 50257
    # latent_id,          # 潜在思维标记的ID <|latent|>, 50259
    end_id,             # 结束潜在思维的特殊标记ID <|end-latent|>, 50258
    no_special_marker=False,     # 是否使用特殊标记 <|start-latent|> 和 <|end-latent|>
    shuffle=False,       # 是否打乱数据集
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # 通常使用当前的 scheduled_stage ，但有一定概率随机选择一个阶段，增加训练的多样性。
        # scheduled_stage 表示当前训练阶段应该使用的推理步骤数量。在代码中，它决定了模型需要跳过多少步骤，以及需要生成多少潜在思维标记(latent tokens)。
        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage
        # 实际上是跳过所有步骤。configs.max_latent_stage可以理解为需要学习的最多的阶段，例如设置为3，代表课程学习3个阶段之后，就可以直接预测完全的latent了
        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            # config里面为True，先设置latent stage的所有token数量等于stage数量，这里等于3。后面再乘上每个latent stage的token数configs.c_thought=2
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        # 代表还在课程学习阶段
        else:   # n_skip_steps ：决定跳过多少推理步骤（这些步骤将由模型生成），n_latent_tokens ：决定使用多少潜在思维标记
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought    # 每个思维的thought个数为configs.c_thought
        # 构建完整的输入序列。sample相当于一个字典，quesion_tokenized这个键包含了问题的分词结果。
        # 这是已经分词的问题文本，例如"计算 5 + 7 的结果是多少？"的标记序列
        # 如果 no_special_marker 为 False （默认情况），则添加一个特殊标记 start_id ，表示潜在思维的开始
        # 这个标记通常是 <|start-latent|> ，用于告诉模型"接下来是潜在思维部分"
        # 重复 latent_id 标记 n_latent_tokens 次，这部分是模型需要学习的潜在思维部分。个数为潜在思维数量乘以每个思维的thought个数config.c_thought
        # 之后就是选取从 n_skip_steps 开始的所有推理步骤
        '''
        假设我们有一个数学问题，包含3个推理步骤，每个步骤被分词后的标记如下：
        sample["steps_tokenized"] = [
            [101, 102, 103],  # 步骤1的标记
            [201, 202, 203],  # 步骤2的标记
            [301, 302, 303]   # 步骤3的标记
        ]
        如果 n_skip_steps = 1 ，那么 sample["steps_tokenized"][n_skip_steps:] 会选取从第二个步骤开始的所有步骤：
        sample["steps_tokenized"][1:] = [
            [201, 202, 203],  # 步骤2的标记
            [301, 302, 303]   # 步骤3的标记
        ]
        然后， itertools.chain.from_iterable 会将这些嵌套的列表展平成一个单一的列表：
        list(itertools.chain.from_iterable(sample["steps_tokenized"][1:])) = [201, 202, 203, 301, 302, 303]
        '''
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # 使用process_dataset方法对base_dataset的每一个sample进行处理
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
