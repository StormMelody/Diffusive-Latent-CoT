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


class DiffusiveCoT_test(nn.Module):
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
        # question,   # [batch_size, 1, 768]
        # steps,      # [batch_size, 7, 768]
        # answer,     # [batch_size, 1, 768]
        # answer_label,
        input_ids,
        labels,
        output_hidden_states: bool = False,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ):

        # Extract arguments from kwargs passed from run.py
        # These are expected to be present based on modifications in run.py
        # Create question_mask, steps_mask, and answer_mask from input_ids
        question_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        steps_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        answer_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

        for i, current_input_ids_sample in enumerate(input_ids):
            # --- Question Mask --- 
            question_start_idx = 0
            for token_idx, token_val in enumerate(current_input_ids_sample):
                if token_val != 50256:  # Padding ID
                    question_start_idx = token_idx
                    break
            else: 
                question_start_idx = len(current_input_ids_sample)

            question_end_idx = len(current_input_ids_sample)
            # Find the end of the question (first 50257 token, which is step_start_id)
            # Search only from question_start_idx onwards
            for token_idx in range(question_start_idx, len(current_input_ids_sample)):
                if current_input_ids_sample[token_idx] == 50257:  # step_start_id
                    question_end_idx = token_idx
                    break
            
            if question_start_idx < question_end_idx:
                question_mask[i, question_start_idx:question_end_idx] = 1

            # --- Steps and Answer Mask --- 
            steps_start_idx = -1
            steps_end_idx = -1
            
            # Find the start of steps (first 50257 token)
            # This can reuse question_end_idx if it was found, as it's the same token
            if question_end_idx < len(current_input_ids_sample) and current_input_ids_sample[question_end_idx] == 50257:
                steps_start_idx = question_end_idx
            else: # Fallback search if question_end_idx wasn't step_start_id
                for token_idx, token_val in enumerate(current_input_ids_sample):
                    if token_val == 50257:  # step_start_id
                        steps_start_idx = token_idx
                        break
            
            if steps_start_idx != -1:
                # Find the end of steps (first 50258 token after steps_start_idx)
                for token_idx in range(steps_start_idx + 1, len(current_input_ids_sample)):
                    if current_input_ids_sample[token_idx] == 50258:  # step_end_id
                        steps_end_idx = token_idx
                        break
                
                if steps_end_idx != -1:
                    steps_mask[i, steps_start_idx:steps_end_idx + 1] = 1 # Include the end_id in steps_mask
                    answer_start_idx = steps_end_idx + 1
                    if answer_start_idx < len(current_input_ids_sample):
                         answer_mask[i, answer_start_idx:] = 1
                else: # If step_end_id is not found, assume steps go to the end
                    steps_mask[i, steps_start_idx:] = 1
        
        # All masks are now created.

        sample_fn = self.base_causallm.forward
        cot_pred = []
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        inputs_embeds = self.embedding(input_ids)
        # Separate embeddings using masks for each item in the batch
        question_embeddings_list = []
        steps_embeddings_list = []
        answer_embeddings_list = []
        answer_labels_list = [] # Initialize list for answer labels
        for i in range(inputs_embeds.size(0)):
            question_embeddings_list.append(inputs_embeds[i][question_mask[i].bool()])
            steps_embeddings_list.append(inputs_embeds[i][steps_mask[i].bool()])
            answer_embeddings_list.append(inputs_embeds[i][answer_mask[i].bool()])
            answer_labels_list.append(labels[i][answer_mask[i].bool()])
        # Pad question_embeddings_list, steps_embeddings_list, answer_embeddings_list
        # 每个list包含了batch_size个tensor, 每个tensor的shape为 (S'_i, D)
        # S'_i 是当前sample中对应部分的非填充token数量, D是embedding维度
        import pdb; pdb.set_trace()
        # Calculate mean embeddings for each part
        question_mean_embeddings_list = torch.stack([torch.mean(emb, dim=0).unsqueeze(0) for emb in question_embeddings_list if emb.numel() > 0], dim=0)

        # Pad steps_embeddings_list, 填充至最大长度，使用 0 向量填充
        if steps_embeddings_list:
            max_steps = 0
            for emb in steps_embeddings_list:
                if emb.numel() > 0:
                    max_steps = max(max_steps, emb.shape[0])
            
            padded_steps_embeddings = []
            for emb in steps_embeddings_list:
                if emb.numel() > 0:
                    pad_len = max_steps - emb.shape[0]
                    if pad_len > 0:
                        # Assuming D is the embedding dimension (emb.shape[1])
                        padding = torch.zeros(pad_len, emb.shape[1], device=emb.device, dtype=emb.dtype)
                        padded_emb = torch.cat([emb, padding], dim=0)
                    else:
                        padded_emb = emb
                    padded_steps_embeddings.append(padded_emb)
                else:
                    # Handle empty tensors if necessary, e.g., by appending a zero tensor of shape [max_steps, D]
                    # For now, let's assume D can be inferred or is fixed. If not, this part needs adjustment.
                    # Example: padded_steps_embeddings.append(torch.zeros(max_steps, question_mean_embeddings_list.shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                    pass # Or handle appropriately
            
            if padded_steps_embeddings: # Ensure list is not empty after processing
                steps_tensor = torch.stack(padded_steps_embeddings, dim=0)
            else:
                # Fallback if all embeddings were empty or filtered out
                # This shape might need to be adjusted based on expected D
                steps_tensor = torch.empty(0, max_steps, question_mean_embeddings_list.shape[-1] if question_mean_embeddings_list.numel() > 0 else 768, device=inputs_embeds.device, dtype=inputs_embeds.dtype) 

        else:
            # Fallback if steps_embeddings_list was initially empty
            steps_tensor = torch.empty(0, 0, question_mean_embeddings_list.shape[-1] if question_mean_embeddings_list.numel() > 0 else 768, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # Pad answer_embeddings_list
        if answer_embeddings_list:
            max_answer_len = 0
            for emb in answer_embeddings_list:
                if emb.numel() > 0:
                    max_answer_len = max(max_answer_len, emb.shape[0])
            
            padded_answer_embeddings = []
            for emb in answer_embeddings_list:
                if emb.numel() > 0:
                    pad_len = max_answer_len - emb.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, emb.shape[1], device=emb.device, dtype=emb.dtype)
                        padded_emb = torch.cat([emb, padding], dim=0)
                    else:
                        padded_emb = emb
                    padded_answer_embeddings.append(padded_emb)
                else:
                    # Handle empty tensors
                    # Assuming D can be inferred from question_mean_embeddings_list or is fixed (e.g., 768)
                    embedding_dim = question_mean_embeddings_list.shape[-1] if question_mean_embeddings_list.numel() > 0 else 768
                    padded_answer_embeddings.append(torch.zeros(max_answer_len, embedding_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype))

            if padded_answer_embeddings: # Ensure list is not empty
                answer_tensor = torch.stack(padded_answer_embeddings, dim=0)
            else:
                # Fallback if all embeddings were empty
                embedding_dim = question_mean_embeddings_list.shape[-1] if question_mean_embeddings_list.numel() > 0 else 768
                answer_tensor = torch.empty(0, max_answer_len, embedding_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        else:
            # Fallback if answer_embeddings_list was initially empty
            embedding_dim = question_mean_embeddings_list.shape[-1] if question_mean_embeddings_list.numel() > 0 else 768
            answer_tensor = torch.empty(0, 0, embedding_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # Pad answer_labels_list (similar logic, but labels are usually LongTensor and padding value is often -100 or tokenizer.pad_token_id)
        if answer_labels_list:
            max_answer_labels_len = 0
            for lbl in answer_labels_list:
                if lbl.numel() > 0:
                    max_answer_labels_len = max(max_answer_labels_len, lbl.shape[0])
            
            padded_answer_labels = []
            for lbl in answer_labels_list:
                if lbl.numel() > 0:
                    pad_len = max_answer_labels_len - lbl.shape[0]
                    if pad_len > 0:
                        # Use -100 for padding labels, common in CrossEntropyLoss
                        padding = torch.full((pad_len,), -100, device=lbl.device, dtype=lbl.dtype) 
                        padded_lbl = torch.cat([lbl, padding], dim=0)
                    else:
                        padded_lbl = lbl
                    padded_answer_labels.append(padded_lbl)
                else:
                    # Handle empty tensors by appending a tensor of padding values
                    padded_answer_labels.append(torch.full((max_answer_labels_len,), -100, device=labels.device, dtype=labels.dtype))
            
            if padded_answer_labels: # Ensure list is not empty
                answer_labels_tensor = torch.stack(padded_answer_labels, dim=0)
            else:
                # Fallback if all labels were empty
                answer_labels_tensor = torch.empty(0, max_answer_labels_len, device=labels.device, dtype=labels.dtype)
        else:
            # Fallback if answer_labels_list was initially empty
            answer_labels_tensor = torch.empty(0, 0, device=labels.device, dtype=labels.dtype)
        
        # import pdb; pdb.set_trace() # Original pdb line, moved for clarity or can be re-inserted as needed
        B, S, D = steps_tensor.shape # Use the shape of the padded and stacked tensor
        # import pdb; pdb.set_trace()
        # 每次循环处理一个step，S为step的步数
        for i in range(S-1):
            timestep = torch.randint(
                0, 
                self.diffusion.num_timesteps, 
                (B,), 
                device=steps_tensor.device
            )
            noise = torch.randn_like(steps_tensor[:, i+1:i+2])  # [B, T, C]
            x = self.diffusion.q_sample(steps_tensor[:, i+1:i+2], timestep, noise)
            model_kwargs = dict(z=question_mean_embeddings_list)
            # 将上一步的step和当前的噪声x拼接起来，再通过mlp降维，作为输入，预测下一步step
            input_embedding = self.adaptor(torch.cat([steps_tensor[:, i:i+1], x], dim=2))
            # DDPM Sampling
            samples = self.diffusion.p_sample_loop(sample_fn, 
                                                    input_embedding.shape, 
                                                    input_embedding, 
                                                    clip_denoised=False,#False, try to set True 
                                                    model_kwargs=model_kwargs,
                                                    progress=False,
                                                    device=steps_tensor.device)

            cot_pred.append(samples["pred_xstart"])
        cot_pred = torch.cat(cot_pred, dim=1)   # shape = [batch_size, 7-1, 768]
        cot_loss = self.mse(cot_pred, steps_tensor[:,1:,...])

        # import pdb; pdb.set_trace()
        past_key_values = samples['kv_cache']
        answer_loss = 0.0
        for i in range(answer_tensor.shape[1]):
            # Check if all labels are -100
            # 好像全 -100 就会nan
            if torch.all(answer_labels_tensor[:, i].view(-1) == -100):
                print(f"Skipping loss calculation for i={i} as all labels are -100")
                break  # or continue, depending on desired behavior
            # import pdb;pdb.set_trace()
            answer_pred = self.base_causallm(
                # inputs_embeds=torch.cat([answer,steps,answer[:,:i]], dim=1),
                inputs_embeds = answer_tensor[:, i-1:i] if i>0 else steps_tensor[:,-1:],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            
            # import pdb; pdb.set_trace()
            #TODO:answer和cot的loss需要用logits而不是embedding计算。
            # answer_loss += self.CEloss(answer_pred['last_hidden_state'][:,i], answer[:,i])
            # answer_loss += self.CEloss(answer_pred['hidden_states'][-1][:,i], answer[:,i])
            answer_loss += self.CEloss(answer_pred['logits'].view(-1, answer_pred['logits'].size(-1)), answer_labels_tensor[:,i].view(-1))
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
