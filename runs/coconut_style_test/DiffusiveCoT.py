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


class DiffusiveCoT(nn.Module):
    def __init__(
        self,
        llm: GPT2LMHeadModel,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        use_diff: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.use_diff = use_diff
        self.base_causallm = llm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        
        self.ddim_diffusion = None
        self.diffusion_steps = 100
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'squaredcos_cap_v2', diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.adaptor  = nn.Linear(2 * 768, 768, bias=True)
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        self.mse = nn.MSELoss()
        self.CEloss = CrossEntropyLoss()

    def forward(
        self, 
        input_ids,
        attention_mask,
        labels,
        position_ids,
        output_hidden_states: bool = False,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ):

        logits = []
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        # 只借鉴了 coconut 的 latent thought 之前的计算过程
        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                # first forward pass，第一步会注意到前面的question部分的token
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],  # 使用 latent之前的部分作为input
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True, # 求输出隐藏状态，用于后续的连续思维生成
                )

                hidden_states_offset = 0
        import pdb; pdb.set_trace()
        sample_fn = self.base_causallm.forward
        cot_pred = []
        B, S, D = steps.shape
        answer_inputs_embeds = self.embedding(answer)   # 通过answer的token_id获取embedding
        # import pdb; pdb.set_trace()
        # 每次循环处理一个step，S为step的步数
        bod_idx = next_compute_range[1] - 1
        bod_last_hidden_state = outputs['hidden_states'][-1][:, bod_idx:bod_idx+1, :]
        for i in range(max_n_latents):
            next_compute_range = (
                next_compute_range[1],  # 上一次结束的位置
                next_compute_range[1] + 1
            )
            timestep = torch.randint(
                0, 
                self.diffusion.num_timesteps, 
                (B,), 
                device=inputs_embeds.device
            )
            noise = torch.randn_like(inputs_embeds[:, next_compute_range[0]:next_compute_range[1]])  # [B, T, C]
            x = self.diffusion.q_sample(inputs_embeds[:, next_compute_range[0]:next_compute_range[1]], timestep, noise)
            model_kwargs = dict(z=bod_last_hidden_state)
            # 将上一步的step和当前的噪声x拼接起来，再通过mlp降维，作为输入，预测下一步step
            input_embedding = self.adaptor(torch.cat([inputs_embeds[:, next_compute_range[0]:next_compute_range[1]], x], dim=2))
            # DDPM Sampling
            samples = self.diffusion.p_sample_loop(sample_fn, 
                                                    input_embedding.shape, 
                                                    input_embedding, 
                                                    clip_denoised=False,#False, try to set True 
                                                    model_kwargs=model_kwargs,
                                                    progress=False,
                                                    device=inputs_embeds.device)
            cot_pred.append(samples["pred_xstart"])
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > i
            ]
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ] 
            hidden_states_offset = next_compute_range[0]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = samples["pred_xstart"][
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                inputs_embeds = torch.stack(
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )
        cot_pred = torch.cat(cot_pred, dim=1) 

        next_compute_range = (next_compute_range[1], input_ids.shape[1])    # 从latent部分一直计算到结束
        # import pdb; pdb.set_trace()
        past_key_values = samples['kv_cache']
        answer_loss = 0.0
        for i in range(answer.shape[1]):
            # import pdb;pdb.set_trace()
            answer_pred = self.base_causallm(
                # inputs_embeds=torch.cat([answer,steps,answer[:,:i]], dim=1),
                inputs_embeds = answer_inputs_embeds[:, i-1:i] if i>0 else steps[:,-1:],
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
        import pdb; pdb.set_trace()
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
