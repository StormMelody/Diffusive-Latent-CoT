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
        use_diff: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.use_diff = use_diff
        self.base_causallm = llm

        self.ddim_diffusion = None
        self.diffusion_steps = 100
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'squaredcos_cap_v2', diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.adaptor  = nn.Linear(2 * 768, 768, bias=True)

        self.mse = nn.MSELoss()
        self.CEloss = CrossEntropyLoss()

    def forward(
        self, 
        question,   # [batch_size, 1, 768]
        steps,      # [batch_size, 7, 768]
        answer,     # [batch_size, 1, 768]
        output_hidden_states: bool = False,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ):
        sample_fn = self.base_causallm.forward
        cot_pred = []
        B, S, D = steps.shape

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
            model_kwargs = dict(z=question)
            # 将steps和x拼接起来，再通过mlp降维，作为输入
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
        for i in range(answer.shape[1]):
            # import pdb;pdb.set_trace()
            answer_pred = self.base_causallm(
                # inputs_embeds=torch.cat([answer,steps,answer[:,:i]], dim=1),
                inputs_embeds = answer[:, i-1:i] if i>0 else steps[:,-1:],
                past_key_values=past_key_values,
                output_hidden_states=False,
            )
            
            import pdb; pdb.set_trace()
            answer_loss += self.CEloss(answer_pred['last_hidden_state'][:,i], answer[:,i])
            past_key_values = answer_pred['past_key_values']

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