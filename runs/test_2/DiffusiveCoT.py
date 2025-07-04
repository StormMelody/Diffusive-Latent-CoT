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
        steps_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device) # Overall steps block
        answer_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

        all_individual_steps_masks = [] # New: list to store lists of individual step masks per sample

        for i, current_input_ids_sample in enumerate(input_ids):
            current_sample_individual_steps_masks = [] # New: list for current sample's individual step masks
            
            # --- Question Mask --- 
            question_start_idx = 0
            # Find first non-padding token for question start
            for token_idx, token_val in enumerate(current_input_ids_sample):
                if token_val != 50256:  # Padding ID
                    question_start_idx = token_idx
                    break
            else: # All padding or empty
                question_start_idx = len(current_input_ids_sample)

            question_end_idx = len(current_input_ids_sample)
            # Find end of question (first 50257 token, which is overall_step_start_id)
            for token_idx in range(question_start_idx, len(current_input_ids_sample)):
                if current_input_ids_sample[token_idx] == 50257:  # overall_step_start_id
                    question_end_idx = token_idx
                    break
            
            if question_start_idx < question_end_idx: # Ensure there's a question part
                question_mask[i, question_start_idx:question_end_idx] = True

            # --- Steps and Answer Mask (Overall + Individual Steps Parsing) --- 
            steps_start_idx = -1  # Start of overall steps block (token 50257)
            steps_end_idx = -1    # End of overall steps block (token 50258)
            
            # Find the start of overall steps (first 50257 token)
            if question_end_idx < len(current_input_ids_sample) and current_input_ids_sample[question_end_idx] == 50257:
                steps_start_idx = question_end_idx
            else: 
                for token_idx_s, token_val_s in enumerate(current_input_ids_sample):
                    if token_val_s == 50257:
                        steps_start_idx = token_idx_s
                        break
            
            if steps_start_idx != -1: # If overall_step_start_id (50257) was found
                # Find the end of overall steps (first 50258 token after steps_start_idx)
                for token_idx_se, token_val_se in enumerate(current_input_ids_sample[steps_start_idx + 1:], start=steps_start_idx + 1):
                    if token_val_se == 50258: # overall_step_end_id
                        steps_end_idx = token_idx_se
                        break
                
                if steps_end_idx != -1: # If 50258 was found
                    steps_mask[i, steps_start_idx : steps_end_idx + 1] = True
                    answer_start_idx = steps_end_idx + 1
                    if answer_start_idx < len(current_input_ids_sample):
                         answer_mask[i, answer_start_idx:] = True
                else: # If 50258 is not found, overall steps go to the end of the sample
                    steps_mask[i, steps_start_idx:] = True
                
                # --- New: Parse Individual Steps (e.g., 16791 ... 198) ---
                INDIVIDUAL_STEP_START_TOKEN_ID = 16791
                INDIVIDUAL_STEP_END_TOKEN_ID = 198

                search_region_start_for_individual_steps = steps_start_idx + 1
                search_region_end_for_individual_steps = \
                    (steps_end_idx - 1) if (steps_end_idx != -1 and steps_end_idx > steps_start_idx) \
                    else (len(current_input_ids_sample) - 1)

                current_parsing_ptr = search_region_start_for_individual_steps
                while current_parsing_ptr <= search_region_end_for_individual_steps:
                    start_token_found_at = -1
                    for k_idx in range(current_parsing_ptr, search_region_end_for_individual_steps + 1):
                        if current_input_ids_sample[k_idx] == INDIVIDUAL_STEP_START_TOKEN_ID:
                            start_token_found_at = k_idx
                            break
                    
                    if start_token_found_at == -1:
                        break

                    end_token_found_at = -1
                    for k_idx in range(start_token_found_at + 1, search_region_end_for_individual_steps + 1):
                        if current_input_ids_sample[k_idx] == INDIVIDUAL_STEP_END_TOKEN_ID:
                            end_token_found_at = k_idx
                            break
                    
                    if end_token_found_at == -1:
                        break 
                    
                    individual_step_mask = torch.zeros_like(current_input_ids_sample, dtype=torch.bool, device=input_ids.device)
                    individual_step_mask[start_token_found_at : end_token_found_at + 1] = True
                    current_sample_individual_steps_masks.append(individual_step_mask)
                    
                    current_parsing_ptr = end_token_found_at + 1
            
            all_individual_steps_masks.append(current_sample_individual_steps_masks)
        
        # All masks are now created.

        sample_fn = self.base_causallm.forward
        cot_pred = []
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        inputs_embeds = self.embedding(input_ids)
        
        # New: Calculate mean embeddings for each individual step
        all_samples_mean_step_embeddings = []
        for i in range(inputs_embeds.size(0)): # Iterate through batch
            sample_mean_step_embeddings = []
            if i < len(all_individual_steps_masks): # Check if masks exist for this sample
                for individual_step_mask in all_individual_steps_masks[i]:
                    # Ensure the mask is boolean and has the same length as the sequence
                    if individual_step_mask.dtype != torch.bool:
                        individual_step_mask = individual_step_mask.bool()
                    if individual_step_mask.shape[0] != inputs_embeds.shape[1]:
                        # This case should ideally not happen if masks are generated correctly
                        # Potentially pad or truncate, or log a warning/error
                        # For now, let's assume correct shape or skip if mismatched severely
                        print(f"Warning: Mismatch in mask shape for sample {i}. Mask: {individual_step_mask.shape}, Embeds: {inputs_embeds.shape[1]}")
                        # If you need to handle this, you might need to resize individual_step_mask
                        # to match inputs_embeds.shape[1], e.g. by padding with False
                        # or by ensuring it's correctly sliced from a full-length template.
                        # For this example, we'll try to apply it if it's shorter, assuming it's a prefix.
                        # This is a simplification and might need robust handling.
                        if individual_step_mask.shape[0] > inputs_embeds.shape[1]:
                           individual_step_mask = individual_step_mask[:inputs_embeds.shape[1]]
                        # If shorter, it might be okay if it's selecting from the start.
                        # However, direct application might fail if shapes don't broadcast or match for selection.
                        # A safer approach is to ensure masks are always full sequence length.
                        # The current mask generation creates full-length masks, so this path is less likely.

                    step_embeddings = inputs_embeds[i][individual_step_mask]
                    if step_embeddings.numel() > 0:
                        mean_step_embedding = torch.mean(step_embeddings, dim=0)
                        sample_mean_step_embeddings.append(mean_step_embedding)
                    else:
                        # Handle case where a step mask is all False (e.g. no step found)
                        # Append a zero vector or a specific placeholder if needed
                        # For now, we can skip or append a zero vector of the correct dimension
                        if inputs_embeds.numel() > 0: # Check if inputs_embeds itself is not empty
                            zero_embedding_dim = inputs_embeds.shape[2]
                            sample_mean_step_embeddings.append(torch.zeros(zero_embedding_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            all_samples_mean_step_embeddings.append(sample_mean_step_embeddings)

        # Determine the maximum number of steps in any sample
        max_steps = 0
        if all_samples_mean_step_embeddings:
            # Ensure all elements are lists of tensors before calculating max_steps
            # And handle cases where a sample might have no steps (empty list)
            for sample_embeddings in all_samples_mean_step_embeddings:
                if isinstance(sample_embeddings, list):
                    max_steps = max(max_steps, len(sample_embeddings))
        
        padded_mean_step_embeddings = []
        # Determine embedding dimension from the first available embedding or inputs_embeds
        embedding_dim = 0
        if inputs_embeds.numel() > 0:
            embedding_dim = inputs_embeds.shape[2]
        else: # Fallback if inputs_embeds is empty, though unlikely if steps were processed
            for sample_embeddings in all_samples_mean_step_embeddings:
                if sample_embeddings and isinstance(sample_embeddings, list) and sample_embeddings[0].numel() > 0:
                    embedding_dim = sample_embeddings[0].shape[0]
                    break
            if embedding_dim == 0 and self.gpt2.transformer.wte.weight.numel() > 0: # Fallback to wte dim
                 embedding_dim = self.gpt2.transformer.wte.weight.shape[1]

        if embedding_dim > 0: # Proceed only if embedding_dim is determined
            for sample_embeddings in all_samples_mean_step_embeddings:
                if not isinstance(sample_embeddings, list):
                    # This case implies an issue upstream or an unexpected structure
                    # For robustness, create a list of empty tensors or handle as an error
                    # Here, we'll assume it should be an empty list of steps if not a list
                    sample_embeddings = [] 

                num_current_steps = len(sample_embeddings)
                padding_needed = max_steps - num_current_steps
                padded_sample_embeddings = list(sample_embeddings) # Make a mutable copy

                if padding_needed > 0:
                    zero_embedding = torch.zeros(embedding_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    padded_sample_embeddings.extend([zero_embedding] * padding_needed)
                
                if not padded_sample_embeddings: # If max_steps is 0 or sample had no steps and max_steps is 0
                    # Handle cases where a sample (and potentially all samples) have no steps.
                    # If max_steps is 0, this loop might not behave as expected or padded_sample_embeddings could be empty.
                    # If max_steps > 0 but this sample is empty, it should have been padded.
                    # If max_steps is 0, we might want to create a placeholder tensor of shape (0, embedding_dim) or similar
                    # For now, if it's empty and max_steps > 0, it means it was padded to max_steps with zeros.
                    # If max_steps is 0, it means no steps anywhere, so we might append an empty tensor or a specific shape.
                    # Let's ensure it's a list of tensors, even if empty, before stacking.
                    # If max_steps is 0, we create a tensor of shape [0, embedding_dim] to allow torch.stack for an empty list of steps
                    if max_steps == 0:
                         padded_mean_step_embeddings.append(torch.empty((0, embedding_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                    else: # Should have been padded if max_steps > 0
                         padded_mean_step_embeddings.append(torch.stack(padded_sample_embeddings)) # Stack if not empty
                else:
                    padded_mean_step_embeddings.append(torch.stack(padded_sample_embeddings)) # Stack if not empty
            
            # Stack all samples' padded mean step embeddings
            if padded_mean_step_embeddings: # Check if the list is not empty
                # Before stacking, ensure all tensors in padded_mean_step_embeddings have the same shape
                # This should be (max_steps, embedding_dim) for each tensor
                # If max_steps is 0, each tensor will be (0, embedding_dim)
                # torch.stack will then create a tensor of shape (batch_size, 0, embedding_dim)
                try:
                    steps_representation = torch.stack(padded_mean_step_embeddings)
                except RuntimeError as e:
                    print(f"Error stacking padded_mean_step_embeddings: {e}")
                    # This might happen if not all samples resulted in a tensor of shape (max_steps, embedding_dim)
                    # For example, if max_steps was 0, and some samples had empty lists that became torch.empty((0, embedding_dim))
                    # while others might have had actual steps. This logic needs to be robust.
                    # If max_steps is 0, all should be (0, dim). If max_steps > 0, all should be (max_steps, dim).
                    # A common issue is if padded_sample_embeddings was empty and torch.stack was called on an empty list.
                    # The current logic tries to avoid this by appending only if padded_sample_embeddings is not empty,
                    # or by appending a specific empty tensor if max_steps is 0.
                    # Fallback: create a zero tensor of expected shape if stacking fails
                    batch_size = inputs_embeds.size(0)
                    steps_representation = torch.zeros((batch_size, max_steps, embedding_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            else: # If padded_mean_step_embeddings is empty (e.g. batch size was 0)
                batch_size = inputs_embeds.size(0) # Should be 0 if all_samples_mean_step_embeddings was empty
                steps_representation = torch.zeros((batch_size, max_steps, embedding_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        else: # embedding_dim could not be determined
            # Handle this case: maybe raise an error or return a specific value
            print("Warning: Embedding dimension could not be determined. Cannot create steps_representation.")
            batch_size = inputs_embeds.size(0)
            # Return a tensor of zeros or None, depending on desired behavior
            steps_representation = torch.zeros((batch_size, 0, 0), device=inputs_embeds.device, dtype=inputs_embeds.dtype) # Or some other appropriate shape
        steps_tensor = steps_representation
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
        # import pdb; pdb.set_trace()
        # Calculate mean embeddings for each part
        question_mean_embeddings_list = torch.stack([torch.mean(emb, dim=0).unsqueeze(0) for emb in question_embeddings_list if emb.numel() > 0], dim=0)

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
        logits = []
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
            logits.append(answer_pred['logits'][:,0]) # Corrected indexing from [:,i] to [:,0]
            #TODO:answer和cot的loss需要用logits而不是embedding计算。
            # answer_loss += self.CEloss(answer_pred['last_hidden_state'][:,i], answer[:,i])
            # answer_loss += self.CEloss(answer_pred['hidden_states'][-1][:,i], answer[:,i])
            answer_loss += self.CEloss(answer_pred['logits'].view(-1, answer_pred['logits'].size(-1)), answer_labels_tensor[:,i].view(-1))
            past_key_values = answer_pred['past_key_values']
        
        if logits:
            logits = torch.stack(logits, dim=1) # Stack along a new dimension (dim=1 for sequence)
        else:
            # Handle the case where logits list is empty, e.g. if the loop was skipped entirely
            # Create an empty tensor or a tensor of zeros with an expected shape if possible
            # This depends on how downstream code uses 'logits'
            # For now, let's create an empty tensor on the correct device
            # Assuming logits would have shape (batch_size, sequence_length, vocab_size)
            # If answer_tensor is available and not empty, we can infer batch_size and vocab_size from answer_pred if it ran at least once
            # Or, more robustly, define a fallback shape or handle it based on model config
            # For simplicity, if answer_pred was populated, use its vocab size.
            # This part might need more context on expected shapes if the loop never runs.
            if 'answer_pred' in locals() and answer_pred['logits'] is not None:
                vocab_size = answer_pred['logits'].size(-1)
                batch_size = answer_tensor.size(0) if answer_tensor is not None else 1 # Default batch_size if not inferable
                logits = torch.empty((batch_size, 0, vocab_size), device=self.device, dtype=self.dtype) # seq_len is 0
            else: # Fallback if answer_pred is not available (e.g. loop for answer_tensor.shape[1] was 0)
                # Try to get device and dtype from model parameters if possible
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                batch_size = answer_tensor.size(0) if answer_tensor is not None and answer_tensor.dim() > 0 else 1
                vocab_size = self.base_causallm.config.vocab_size # Assuming base_causallm has a standard config
                logits = torch.empty((batch_size, 0, vocab_size), device=device, dtype=dtype)

        loss = cot_loss + answer_loss
        # return cot_loss + answer_loss
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor, # Changed from question_ids to input_ids
        tokenizer: GPT2Tokenizer, # Or any compatible tokenizer
        attention_mask: Optional[torch.Tensor] = None, # Added attention_mask
        max_length: int = 50,
        max_new_tokens=64,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        use_ddim: bool = False, # For diffusion model sampling
        num_ddim_steps: int = 10, # For diffusion model sampling
        **kwargs # Added to catch other arguments
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Extract question_ids from input_ids similar to the forward method
        question_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        for i, current_input_ids_sample in enumerate(input_ids):
            question_start_idx = 0
            for token_idx, token_val in enumerate(current_input_ids_sample):
                if token_val != 50256:  # Padding ID
                    question_start_idx = token_idx
                    break
            else: 
                question_start_idx = len(current_input_ids_sample)

            question_end_idx = len(current_input_ids_sample)
            for token_idx in range(question_start_idx, len(current_input_ids_sample)):
                if current_input_ids_sample[token_idx] == 50257:  # overall_step_start_id
                    question_end_idx = token_idx
                    break
            
            if question_start_idx < question_end_idx:
                question_mask[i, question_start_idx:question_end_idx] = True

        # Create a list of question_ids tensors
        question_ids_list = []
        for i in range(input_ids.size(0)):
            current_question_ids = input_ids[i][question_mask[i]]
            if current_question_ids.numel() == 0: # Handle empty question case
                # Fallback: use a single padding token or handle as an error
                # For now, let's use a single pad token if tokenizer is available
                pad_token_for_empty_q = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256
                current_question_ids = torch.tensor([pad_token_for_empty_q], device=device, dtype=torch.long)
            question_ids_list.append(current_question_ids)

        question_ids_padded = pad_sequence(question_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256)

        # 1. Get question embeddings
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        question_embeds = self.embedding(question_ids_padded) # (batch_size, padded_seq_len, embed_dim)
        actual_lengths = torch.tensor([len(q_ids) for q_ids in question_ids_list], device=device)
        padding_mask_for_mean = torch.arange(question_ids_padded.size(1), device=device)[None, :] < actual_lengths[:, None]
        
        # Expand padding_mask_for_mean to match question_embeds dimensions for masked_fill
        expanded_padding_mask = padding_mask_for_mean.unsqueeze(-1).expand_as(question_embeds)

        masked_question_embeds = question_embeds.masked_fill(~expanded_padding_mask, 0.0)
        sum_question_embed = torch.sum(masked_question_embeds, dim=1) # Sum non-padded embeddings
        mean_question_embed = sum_question_embed / actual_lengths.unsqueeze(1).clamp(min=1) # Divide by actual lengths
        # mean_question_embed = torch.mean(question_embeds, dim=1) # (batch_size, embed_dim) - Original, less accurate with padding

        if self.use_diff and self.diffusion is not None:
            max_cot_length = 16  # Define a reasonable default for CoT length

            class Denoiser(nn.Module):
                def __init__(self, model, diffusion, question_embed):
                    super().__init__()
                    self.model = model
                    self.diffusion = diffusion
                    self.question_embed = question_embed

                def forward(self, x, t, **kwargs):
                    inputs_embeds = kwargs.get('inputs_embeds', x)
                    time_embed = self.diffusion.time_embedding(t, inputs_embeds.shape[-1])
                    conditioned_input = inputs_embeds + time_embed.unsqueeze(1) + self.question_embed.unsqueeze(1)
                    output = self.model(inputs_embeds=conditioned_input, return_dict=True)
                    return output.last_hidden_state

            denoiser = Denoiser(self.base_causallm, self.diffusion, mean_question_embed)
            shape = (batch_size, max_cot_length, self.embedding.embedding_dim)

            generated_steps_representation = self.diffusion.p_sample_loop(
                denoiser,
                shape,
                clip_denoised=False,
                progress=True,
            )

        else:
            generated_steps_representation = torch.zeros(batch_size, 1, self.embedding.embedding_dim, device=device)


        if generated_steps_representation.dim() == 3 and generated_steps_representation.size(1) > 0:
            summarized_cot_embed = torch.mean(generated_steps_representation, dim=1) # (batch_size, embed_dim)
        elif generated_steps_representation.dim() == 2: # If it's already (batch_size, embed_dim)
            summarized_cot_embed = generated_steps_representation
        else: # Handle empty or unexpected shape
            summarized_cot_embed = torch.zeros(batch_size, self.embedding.embedding_dim, device=device)

        # 5. Combine mean_question_embed and summarized_cot_embed using the adaptor
        combined_embed = torch.cat([mean_question_embed, summarized_cot_embed], dim=1) # (batch_size, 2 * embed_dim)
        adaptor_output = self.adaptor(combined_embed) # (batch_size, embed_dim)

        final_input_embeds = adaptor_output.unsqueeze(1) # Shape: (batch_size, 1, embed_dim)

        attention_mask = torch.ones(final_input_embeds.size(0), final_input_embeds.size(1), device=device, dtype=torch.long)


        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id


        adaptor_input = torch.cat([mean_question_embed, summarized_cot_embed], dim=-1)
        combined_representation = self.adaptor(adaptor_input) # (batch_size, embed_dim)

        inputs_embeds_for_generation = combined_representation.unsqueeze(1) # Shape: (batch_size, 1, embed_dim)

        final_attention_mask = torch.ones(inputs_embeds_for_generation.size(0), 1, device=inputs_embeds_for_generation.device, dtype=torch.long)

        # 6. Generate answer autoregressively using the base causal LM
        generated_ids = self.base_causallm.generate(
            inputs_embeds=inputs_embeds_for_generation, # Corrected variable
            attention_mask=final_attention_mask, # Use the newly constructed attention mask
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            # early_stopping=True # Optional: if you want to stop early for beams
        )

        decoded_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded_answers, generated_ids, generated_steps_representation # Return decoded text, raw IDs, and CoT

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
