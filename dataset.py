# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import List, Dict, Union, Optional # Added Optional as it was in the previous version of the file

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
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
    # import pdb; pdb.set_trace()
    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    start_id: Optional[int] = None  # Added
    end_id: Optional[int] = None    # Added
    label_pad_token_id: Optional[int] = -100

    def __init__(self, tokenizer: PreTrainedTokenizerBase, latent_id: Optional[int] = None, 
                 start_id: Optional[int] = None, end_id: Optional[int] = None, 
                 label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.latent_id = latent_id
        self.start_id = start_id
        self.end_id = end_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], int]]], return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        # Align start_id to be at the same position for all samples in the batch
        if self.start_id is not None:
            indices_of_start_id = []
            for feature in features:
                try:
                    indices_of_start_id.append(feature["input_ids"].index(self.start_id))
                except ValueError:
                    # If start_id is not in this feature, we might skip it or assign a default (e.g., -1 or len)
                    # For now, let's assume start_id should be present if self.start_id is not None.
                    # If it can be legitimately absent, this logic needs adjustment.
                    pass # Or raise an error, or handle as per requirements
            
            if indices_of_start_id: # Only proceed if start_id was found in at least one feature
                latest_earliest_start_id = max(indices_of_start_id)
                for feature in features:
                    try:
                        current_start_id_index = feature["input_ids"].index(self.start_id)
                        n_tok_pad_for_start_id = latest_earliest_start_id - current_start_id_index
                    except ValueError:
                        # If start_id is not in this specific feature (but was in others),
                        # pad it as if its start_id was at the beginning (index 0) relative to latest_earliest_start_id.
                        # This case needs careful consideration based on expected data structure.
                        # Assuming for now if start_id is expected, it should be there.
                        # If it's truly optional per sample, this padding might be incorrect.
                        # A safer default might be to pad up to latest_earliest_start_id if start_id is missing.
                        # For this implementation, we'll assume if self.start_id is set, it's expected in all relevant features.
                        # If a feature doesn't have start_id, we skip padding for it here, or one might pad it by latest_earliest_start_id.
                        # Let's assume if start_id is not found, we don't pad this specific feature based on start_id alignment.
                        n_tok_pad_for_start_id = 0

                    if n_tok_pad_for_start_id > 0:
                        if "position_ids" not in feature or feature["position_ids"] is None:
                            feature["position_ids"] = list(range(len(feature["input_ids"])))
                        feature["position_ids"] = [0] * n_tok_pad_for_start_id + feature["position_ids"]
                        
                        feature["input_ids"] = [50256] * n_tok_pad_for_start_id + feature["input_ids"] # Changed to use 50256 for padding
                        
                        if "labels" in feature and feature["labels"] is not None:
                            feature["labels"] = [self.label_pad_token_id] * n_tok_pad_for_start_id + feature["labels"]
                        
                        if "attention_mask" not in feature or feature["attention_mask"] is None:
                            # Initialize attention_mask based on the length *before* this start_id padding
                            # This assumes attention_mask corresponds to original input_ids length if not present
                            # However, input_ids was already modified if latent_id padding happened before. This needs care.
                            # For safety, let's assume attention_mask should exist or be derived from current input_ids length *before* this padding.
                            # Given this runs before latent_id padding, this is safer.
                            original_len = len(feature["input_ids"]) - n_tok_pad_for_start_id # Length before we added pads
                            feature["attention_mask"] = [1] * original_len
                        
                        feature["attention_mask"] = [1] * n_tok_pad_for_start_id + feature["attention_mask"] # Changed to use 1 for attention_mask padding

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        # Align latent_id (if present and configured) to be at the same position
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id is not None and self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id is not None and self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0 # Should not happen if latent_id is in input_ids, but defensive
                
                # Ensure position_ids is initialized if not present
                if "position_ids" not in feature:
                    feature["position_ids"] = list(range(len(feature["input_ids"])))

                feature["position_ids"] = [0] * n_tok_pad + feature["position_ids"]
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                
                # Ensure attention_mask is initialized if not present
                if "attention_mask" not in feature:
                    feature["attention_mask"] = [1] * len(feature["input_ids"]) # Before padding

                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        if return_tensors is None:
            return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        keys_to_pad_by_tokenizer = ["input_ids", "attention_mask"]
        auxiliary_keys = ["question_tokenized", "steps_tokenized", "answer_tokenized", "idx"]

        main_features_to_pad = []
        auxiliary_data: Dict[str, List] = {key: [] for key in auxiliary_keys if key in features[0]}

        for feature in features:
            current_main_feature = {}
            for k, v in feature.items():
                if k in keys_to_pad_by_tokenizer:
                    current_main_feature[k] = v
                elif k in auxiliary_keys and k in feature: # Check if k in feature
                    auxiliary_data[k].append(v)
            main_features_to_pad.append(current_main_feature)
        
        # This function pad_without_fast_tokenizer_warning is not standard.
        # Assuming it's defined elsewhere in your codebase or you have a similar utility.
        # For this example, I'll proceed as if it correctly pads main_features_to_pad.
        # If it's not available, you might need to replace this with standard tokenizer.pad
        if hasattr(self.tokenizer, 'pad_without_fast_tokenizer_warning'): # Check if custom method exists
             batch = self.tokenizer.pad_without_fast_tokenizer_warning(
                 main_features_to_pad, 
                 padding=True,
                 pad_to_multiple_of=None,
                 return_tensors=return_tensors,
             )
        else: # Fallback to standard pad
             batch = self.tokenizer.pad(
                 main_features_to_pad, 
                 padding=True,
                 pad_to_multiple_of=None,
                 return_tensors=return_tensors,
             )

        # Add auxiliary data to the batch as is (mostly as lists of lists or lists of items)
        for key, value_list in auxiliary_data.items():
            if value_list: 
                batch[key] = value_list

        # NEW: Process steps_tokenized
        if "steps_tokenized" in batch and batch["steps_tokenized"]:
            if self.start_id is None or self.end_id is None:
                raise ValueError("start_id and end_id must be provided to MyCollator for processing steps_tokenized")
            
            processed_steps_sequences = []
            # batch["steps_tokenized"] is List[List[List[int]]]
            # Each element is all steps for one sample: List[List[int]]
            for steps_sequences_for_sample in batch["steps_tokenized"]:
                # Flatten the list of lists of tokens for the current sample
                flat_steps_for_sample = []
                if isinstance(steps_sequences_for_sample, list): # Ensure it's a list
                    for step_sequence in steps_sequences_for_sample:
                        if isinstance(step_sequence, list): # Ensure inner element is also a list
                            flat_steps_for_sample.extend(step_sequence)
                        elif isinstance(step_sequence, int): # Handle case where it might be already flat for some reason
                            flat_steps_for_sample.append(step_sequence)
                        # else: skip or log unexpected type
                
                # Prepend start_id and append end_id
                processed_steps_sequences.append([self.start_id] + flat_steps_for_sample + [self.end_id])
            
            max_steps_len = 0
            if processed_steps_sequences:
                max_steps_len = max(len(s) for s in processed_steps_sequences)

            padded_steps_list = []
            for s_seq in processed_steps_sequences:
                padding_length = max_steps_len - len(s_seq)
                padded_steps_list.append(s_seq + [self.tokenizer.pad_token_id] * padding_length)
            
            if padded_steps_list:
                batch["steps_ids_padded"] = torch.tensor(padded_steps_list, dtype=torch.int64)
            elif batch["steps_tokenized"]: # Original list was not empty
                batch_size = len(batch["steps_tokenized"])
                batch["steps_ids_padded"] = torch.empty((batch_size, 0), dtype=torch.int64)
            # If batch["steps_tokenized"] was empty, steps_ids_padded is not added, which is fine.

        # NEW: Process answer_tokenized
        if "answer_tokenized" in batch and batch["answer_tokenized"]:
            processed_answers_sequences = []
            for answer_list_per_sample in batch["answer_tokenized"]:
                processed_answers_sequences.append(answer_list_per_sample + [self.tokenizer.eos_token_id])

            max_answers_len = 0
            if processed_answers_sequences:
                max_answers_len = max(len(a) for a in processed_answers_sequences)

            padded_answers_list = []
            for a_seq in processed_answers_sequences:
                padding_length = max_answers_len - len(a_seq)
                padded_answers_list.append(a_seq + [self.tokenizer.pad_token_id] * padding_length)

            padded_answer_labels_list = [] # For answer_labels_padded
            if padded_answers_list:
                batch["answer_ids_padded"] = torch.tensor(padded_answers_list, dtype=torch.int64)
                # Create answer_labels_padded
                for a_seq in processed_answers_sequences: # Use processed_answers_sequences before pad_token_id
                    padding_length = max_answers_len - len(a_seq)
                    padded_answer_labels_list.append(a_seq + [self.label_pad_token_id] * padding_length)
                if padded_answer_labels_list:
                    batch["answer_labels_padded"] = torch.tensor(padded_answer_labels_list, dtype=torch.int64)

            elif batch["answer_tokenized"]: # If original answer_tokenized was not empty but processed_answers_sequences became empty (edge case)
                batch_size = len(batch["answer_tokenized"])
                batch["answer_ids_padded"] = torch.empty((batch_size, 0), dtype=torch.int64)
                batch["answer_labels_padded"] = torch.empty((batch_size, 0), dtype=torch.int64)

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        
        position_ids_list = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys() and features[0]["position_ids"] is not None
            else None
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels) if labels else 0
            padded_labels = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.int64)

        if position_ids_list is not None:
            max_pos_length = max(len(l) for l in position_ids_list) if position_ids_list else 0
            padded_position_ids = [
                pos_id_list + [0] * (max_pos_length - len(pos_id_list))
                for pos_id_list in position_ids_list
            ]
            batch["position_ids"] = torch.tensor(
                padded_position_ids, dtype=torch.int64
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
    latent_id,          # 潜在思维标记的ID <|latent|>, 50259
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
        if configs.coconut:
            all_steps_tokenized = list(itertools.chain.from_iterable(sample["steps_tokenized"]))
            
            input_ids = (
                sample["question_tokenized"]
                + [start_id]
                + all_steps_tokenized
                + [end_id]
                + sample["answer_tokenized"]
            )
            
            labels = (
                [-100] * len(sample["question_tokenized"])
                + [-100]  # Mask start_id for CoT steps
                + all_steps_tokenized # These are the actual tokens for steps, used as labels
                + [-100]  # Mask end_id for CoT steps
                + sample["answer_tokenized"] # These are the actual tokens for answer, used as labels
            )
        else: # Original logic for Coconut (non-CoT)
            steps_to_include = list(itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:]))
            
            input_ids = (
                sample["question_tokenized"]
                + ([] if no_special_marker else [start_id])
                + [latent_id] * n_latent_tokens
                + ([] if no_special_marker else [end_id])
                + steps_to_include
                + sample["answer_tokenized"]
            )

            labels = (
                [-100] * (
                    len(sample["question_tokenized"])
                    + n_latent_tokens
                    + n_additional_tokens
                )
                + steps_to_include
                + sample["answer_tokenized"]
            )
        
        # Ensure lengths match
        assert len(input_ids) == len(labels), f"Input IDs length {len(input_ids)} != Labels length {len(labels)}. Sample idx: {sample.get('idx', 'Unknown')}"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "idx": sample["idx"],
            "position_ids": list(range(len(input_ids))),
            "question_tokenized": sample["question_tokenized"],
            "steps_tokenized": sample["steps_tokenized"],
            "answer_tokenized": sample["answer_tokenized"],
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
