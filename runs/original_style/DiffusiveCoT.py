import sys
sys.path.append("/data2/xxw_data/projects/LLM/Diffusive-Latent-CoT/models/vlas")
import torch
import torch.nn as nn
import numpy as np
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizerFast
from typing import Optional
from models.backbones import LLMBackbone, VisionBackbone
from models.vlms import PrismaticVLM
from models.diffusion import create_diffusion
from util import FusedMLPProjector, LinearProjector, MLPProjector
from overwatch import initialize_overwatch
from vla import ActionTokenizer
from transformers import GPT2Tokenizer, GPT2Model
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel
from collections import namedtuple
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
class DiffusiveCoT(nn.Module):
    def __init__(
        self,
        llm: GPT2LMHeadModel,
        use_diff: bool = True,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.use_diff = use_diff
        self.base_causallm = llm

        self.ddim_diffusion = None
        self.diffusion_steps = 100
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'squaredcos_cap_v2', diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        hidden_size = self.base_causallm.config.hidden_size
        self.adaptor  = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            # embedding为词嵌入层，能够通过token_id获取对应的embedding
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        self.mse = nn.MSELoss()
        self.CEloss = CrossEntropyLoss()

    def forward(
        self, 
        input_ids,
        labels,
        output_hidden_states: bool = False,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ):
        masks = parse_input_ids(input_ids)
        inputs_embeds = self.embedding(input_ids)

        # 根据mask中的值对embedding和id进行分类
        # len(classified_ids_list) = batch_size
        # classified_ids_list[0].keys() = dict_keys([1, 2, 3, 4, 5, 6, 7, 8])
        classified_embeds_list = []
        classified_ids_list = []
        for embed, ids, mask in zip(inputs_embeds, input_ids, masks):
            classified_embeds = {}
            classified_ids = {}
            # 获取mask中所有唯一的非零类别值
            unique_values = torch.unique(mask)
            for val in unique_values:
                if val == 0: # 忽略值为0的背景/填充部分
                    continue
                # 找到当前类别值在mask中的所有位置
                class_indices = (mask == val)
                # 提取对应位置的embedding和id
                classified_embeds[val.item()] = embed[class_indices]
                classified_ids[val.item()] = ids[class_indices]
            classified_embeds_list.append(classified_embeds)
            classified_ids_list.append(classified_ids)
        
        # 提取 question embedding
        question_embeds_list = []
        for classified_embeds in classified_embeds_list:
            # 确保key=1存在，并且对应的embedding不为空
            if 1 in classified_embeds and classified_embeds[1].numel() > 0:
                question_embeds_list.append(classified_embeds[1])
        # 往 question embedding 最前面填充 self.pad_token_id
        if question_embeds_list:
            # 使用 pad_sequence 和 flip 实现左填充
            # 1. 为了实现左填充，我们先反转序列
            reversed_question_embeds_list = [torch.flip(embed, dims=[0]) for embed in question_embeds_list]
            # 2. 使用 pad_sequence 进行右填充 (用 0)
            padded_reversed_embeds = torch.nn.utils.rnn.pad_sequence(reversed_question_embeds_list, batch_first=True, padding_value=0)
            # 3. 再反转回来，实现左填充
            question_embeds = torch.flip(padded_reversed_embeds, dims=[1])
            
            # 4. 获取 pad_token_id 对应的 embedding
            pad_token_id = self.pad_token_id
            pad_embed = self.base_causallm.get_input_embeddings()(torch.tensor(pad_token_id, device=inputs_embeds.device))

            # 5. 创建一个 mask 来标识填充位置，并将 0 替换为 pad_embed
            # 首先，创建每个序列的 mask (1 for real tokens, 0 for padding)
            masks = [torch.ones(embed.shape[0], device=inputs_embeds.device) for embed in question_embeds_list]
            # 同样的方式进行 padding
            reversed_masks = [torch.flip(mask, dims=[0]) for mask in masks]
            padded_reversed_masks = torch.nn.utils.rnn.pad_sequence(reversed_masks, batch_first=True, padding_value=0)
            attention_mask = torch.flip(padded_reversed_masks, dims=[1])

            # 使用 mask 将填充位置的 embedding 替换为 pad_embed
            question_embeds = question_embeds * attention_mask.unsqueeze(-1) + (1 - attention_mask.unsqueeze(-1)) * pad_embed.view(1, 1, -1)
        else:
            # 如果列表为空，则创建一个空的张量
            question_embeds = torch.empty(0, 0, inputs_embeds.shape[-1], device=inputs_embeds.device)

        # 获取answer embedding和id，并填充为三维张量
        answer_embeds_list = []
        answer_ids_list = []
        for classified_embeds, classified_ids in zip(classified_embeds_list, classified_ids_list):
            if 2 in classified_embeds and classified_embeds[2].numel() > 0:
                answer_embeds_list.append(classified_embeds[2])
                answer_ids_list.append(classified_ids[2])
            else:
                # For empty answers, add an empty tensor
                answer_embeds_list.append(torch.empty(0, inputs_embeds.shape[-1], device=inputs_embeds.device))
                answer_ids_list.append(torch.empty(0, dtype=torch.long, device=input_ids.device))

        # 使用 pad_sequence 进行填充，使得长度对齐
        answer_embeds = torch.nn.utils.rnn.pad_sequence(answer_embeds_list, batch_first=True, padding_value=0.0)
        answer_label = torch.nn.utils.rnn.pad_sequence(answer_ids_list, batch_first=True, padding_value=-100)

        # 创建一个 mask 来识别填充的位置，然后使用<endoftext>来替代原来的0
        original_lengths = [embed.shape[0] for embed in answer_embeds_list]
        if answer_embeds.numel() > 0:
            max_len = answer_embeds.shape[1]
            # (batch_size, max_len)
            padding_mask = torch.arange(max_len, device=inputs_embeds.device)[None, :] >= torch.tensor(original_lengths, device=inputs_embeds.device)[:, None]

            # 获取 <|endoftext|> token (ID 50256) 的 embedding
            pad_token_id = self.pad_token_id
            pad_embedding = self.embedding(torch.tensor([pad_token_id], device=inputs_embeds.device)).squeeze(0)

            # 使用 mask 将填充位置的值替换为 pad_embedding
            answer_embeds[padding_mask] = pad_embedding

        # 处理步骤序列（BOD + steps + EOD），并进行填充
        from torch.nn.utils.rnn import pad_sequence

        sequences_to_pad = []
        for classified_embeds in classified_embeds_list:
            sequence = []

            # 1. 添加BOD嵌入 (key=3)
            if 3 in classified_embeds and classified_embeds[3].numel() > 0:
                sequence.append(classified_embeds[3].mean(dim=0))
            else:
                sequence.append(torch.zeros(inputs_embeds.shape[-1], device=inputs_embeds.device))

            # 2. 添加每个步骤的平均嵌入 (key>=6)
            step_keys = sorted([k for k in classified_embeds.keys() if isinstance(k, int) and k >= 6])
            for step_key in step_keys:
                if classified_embeds[step_key].numel() > 0:
                    sequence.append(classified_embeds[step_key].mean(dim=0))
            
            # 3. 添加EOD嵌入 (key=4)
            if 4 in classified_embeds and classified_embeds[4].numel() > 0:
                sequence.append(classified_embeds[4].mean(dim=0))
            else:
                sequence.append(torch.zeros(inputs_embeds.shape[-1], device=inputs_embeds.device))

            sequences_to_pad.append(torch.stack(sequence, dim=0))

        # 4. 填充序列使其长度一致
        step_sequence_embeds = pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
        steps = step_sequence_embeds
        # import pdb; pdb.set_trace()
        sample_fn = self.base_causallm.forward
        cot_pred = []
        B, S, D = steps.shape

        # import pdb; pdb.set_trace()
        # 每次循环处理一个step，S为step的步数
        for i in range(S-1):
            timestep = torch.randint(
                0, 
                self.diffusion.num_timesteps, 
                (B,), 
                device=steps.device
            )
            noise = torch.randn_like(steps[:, i+1:i+2])  # [B, T, C]
            x = self.diffusion.q_sample(steps[:, i+1:i+2], timestep, noise)
            model_kwargs = dict(z=question_embeds)
            # 将上一步的step和当前的噪声x拼接起来，再通过mlp降维，作为输入，预测下一步step
            input_embedding = self.adaptor(torch.cat([steps[:, i:i+1], x], dim=2))
            # DDPM Sampling
            samples = self.diffusion.p_sample_loop(sample_fn, 
                                                    input_embedding.shape, 
                                                    input_embedding, 
                                                    clip_denoised=False,#False, try to set True 
                                                    model_kwargs=model_kwargs,
                                                    progress=False,
                                                    device=steps.device)

            cot_pred.append(samples["pred_xstart"])
        cot_pred = torch.cat(cot_pred, dim=1)   # shape = [batch_size, 7-1, 768]
        cot_loss = self.mse(cot_pred, steps[:,1:,...])

        # import pdb; pdb.set_trace()
        past_key_values = samples['kv_cache']
        answer_loss = 0.0
        for i in range(answer_embeds.shape[1]):
            # import pdb;pdb.set_trace()
            answer_pred = self.base_causallm(
                # inputs_embeds=torch.cat([answer,steps,answer[:,:i]], dim=1),
                inputs_embeds = answer_embeds[:, i-1:i] if i>0 else steps[:,-1:],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            
            # import pdb; pdb.set_trace()
            #TODO:answer和cot的loss需要用logits而不是embedding计算。
            # answer_loss += self.CEloss(answer_pred['last_hidden_state'][:,i], answer[:,i])
            # answer_loss += self.CEloss(answer_pred['hidden_states'][-1][:,i], answer[:,i])
            answer_loss += self.CEloss(
                answer_pred['logits'].view(-1, answer_pred['logits'].size(-1)),
                answer_label[:,i].view(-1)
                )
            past_key_values = answer_pred['past_key_values']
        # import pdb; pdb.set_trace()
        return cot_loss + answer_loss

    def _repeat_tensor(
        self, 
        tensor: Optional[torch.Tensor], 
        repeated_diffusion_steps: int
    ) -> Optional[torch.Tensor]:
        """
        Repeat a tensor along the first dimension

        Args:
            tensor: Input tensor to repeat
            repeated_diffusion_steps: Number of times to repeat

        Returns:
            Repeated tensor or None
        """
        if tensor is None:
            return None
        return tensor.repeat(repeated_diffusion_steps, *([1] * (tensor.ndimension() - 1)))

def parse_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Parses a tensor of token IDs into a mask tensor indicating the position of
    question, answer, BOD, EOD, and steps.

    Args:
        input_ids: A 1D or 2D tensor of integer token IDs.

    Returns:
        A mask tensor of the same shape as input_ids, where:
        - 1: question
        - 2: answer
        - 3: BOD
        - 4: EOD
        - 5: END_OF_TEXT_TOKEN
        - 6, 7, 8, ...: steps
    """
    # BOD_TOKEN = 50257
    # EOD_TOKEN = 50258
    # STEP_START_TOKEN = 16791
    # STEP_END_TOKEN = 198
    # END_OF_TEXT_TOKEN = 50256
    BOD_TOKEN = 128256
    EOD_TOKEN = 128257
    STEP_START_TOKEN = 2501
    STEP_END_TOKEN = 40171
    END_OF_TEXT_TOKEN = 128009
    is_1d = input_ids.dim() == 1
    if is_1d:
        input_ids_2d = input_ids.unsqueeze(0)
    else:
        input_ids_2d = input_ids

    mask = torch.zeros_like(input_ids_2d)
    batch_size = input_ids_2d.shape[0]

    # Find BOD and EOD tokens
    bod_nz = (input_ids_2d == BOD_TOKEN).nonzero(as_tuple=True)
    eod_nz = (input_ids_2d == EOD_TOKEN).nonzero(as_tuple=True)

    # Fallback if tokens are not found in any sequence
    if len(bod_nz[0]) == 0 or len(eod_nz[0]) == 0:
        return torch.ones_like(input_ids)

    bod_indices = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
    eod_indices = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)

    # Get the first occurrence of BOD/EOD for each batch item
    bod_batch_idx, bod_seq_idx = bod_nz
    unique_bod_batch = torch.unique(bod_batch_idx)
    for i in unique_bod_batch:
        positions = bod_seq_idx[bod_batch_idx == i]
        if positions.numel() > 0:
            bod_indices[i] = positions[0]

    eod_batch_idx, eod_seq_idx = eod_nz
    unique_eod_batch = torch.unique(eod_batch_idx)
    for i in unique_eod_batch:
        positions = eod_seq_idx[eod_batch_idx == i]
        if positions.numel() > 0:
            eod_indices[i] = positions[0]
    
    # Handle sequences where tokens might be missing
    valid_mask = (bod_indices != -1) & (eod_indices != -1)
    if not valid_mask.all():
        mask[~valid_mask] = 1 # Fallback for invalid sequences

    # Process valid sequences
    for i in range(batch_size):
        if valid_mask[i]:
            bod_idx = bod_indices[i]
            eod_idx = eod_indices[i]
            
            mask[i, :bod_idx] = 1   # question
            mask[i, bod_idx] = 3    # bod
            mask[i, eod_idx] = 4    # eod
            mask[i, eod_idx + 1:] = 2   # answer

    # Find and mark steps, overwriting the general thought mask
    step_start_nz = (input_ids_2d == STEP_START_TOKEN).nonzero(as_tuple=True)
    step_end_nz = (input_ids_2d == STEP_END_TOKEN).nonzero(as_tuple=True)

    if len(step_start_nz[0]) > 0 and len(step_end_nz[0]) > 0:
        start_batch_indices, step_starts = step_start_nz
        end_batch_indices, step_ends = step_end_nz
        for i in range(batch_size):
            if valid_mask[i]:
                bod_idx = bod_indices[i]
                eod_idx = eod_indices[i]

                current_starts = step_starts[start_batch_indices == i]
                current_ends = step_ends[end_batch_indices == i]
                
                # Filter steps to be within thought
                valid_steps_mask_of_current_starts = (current_starts > bod_idx) & (current_starts < eod_idx)
                valid_steps_mask_of_current_ends = (current_ends > bod_idx) & (current_ends < eod_idx)
                current_starts = current_starts[valid_steps_mask_of_current_starts]
                current_ends = current_ends[valid_steps_mask_of_current_ends]
                
                # Ensure we have paired start and end tokens
                num_steps = min(len(current_starts), len(current_ends))
                current_starts = current_starts[:num_steps]
                current_ends = current_ends[:num_steps]
                step_mask_value = 6
                for start, end in zip(current_starts, current_ends):
                    if start < end: # Basic sanity check
                        mask[i, start:end + 1] = step_mask_value
                        step_mask_value += 1
                        
    
    if is_1d:
        mask = mask.squeeze(0)
    if not is_1d:
        mask[input_ids_2d == END_OF_TEXT_TOKEN] = 5
    return mask