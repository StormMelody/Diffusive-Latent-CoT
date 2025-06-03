import torch
import torch.nn as nn
import numpy as np
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizerFast

from models.backbones import LLMBackbone, VisionBackbone
from models.vlms import PrismaticVLM
from models.diffusion import create_diffusion
from util import FusedMLPProjector, LinearProjector, MLPProjector
from overwatch import initialize_overwatch
from vla import ActionTokenizer

class DiffusiveCoT(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_tokenizer: ActionTokenizer,
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        use_diff: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.action_tokenizer = action_tokenizer

        self.use_diff = use_diff
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.vlm.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.all_module_keys=[]
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        self.norm_stats = norm_stats
        self._trainable_module_keys = []

        if self.use_diff:
            self.ddim_diffusion = None
            self.diffusion_steps = 100
            self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'squaredcos_cap_v2', diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
    
    def forward(
        self, 
        instruction: str = "", 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 0.0, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        action_dim: int = 7,
        cur_robot_state: Optional[str] = None,
        multi_view: bool = True,
        predict_mode: str = "diff+ar",
        **kwargs: str
    ):
        tokenizer = self.vlm.llm_backbone.tokenizer
        device = self.vlm.device
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        
        message = f"What action should the robot take to {instruction.lower()}?"
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=message)
        prompt_text = prompt_builder.get_prompt()
        
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(device)
        
        if not isinstance(tokenizer, LlamaTokenizerFast):
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        
        def append_tokens(ids_to_append):
            token_tensor = torch.tensor([ids_to_append], dtype=torch.long, device=device)
            return torch.cat((input_ids, token_tensor), dim=1)
        
        has_empty_token = lambda: torch.all(input_ids[:, -1] == 29871)
        
        if self.vlm.model_id == 'prism-dinosiglip-224px+7b':
            if not has_empty_token():
                input_ids = append_tokens([29871, 32001, 32002, 29871])
        elif self.vlm.model_id == 'phi-2+3b':
            input_ids = append_tokens([220, 50296, 50297])
        else:
            raise ValueError(f"Unsupported predict_mode = {predict_mode}")
        
        if cur_robot_state is not None:
            proprio_norm_stats = self.get_proprio_stats(unnorm_key)
            mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(proprio_norm_stats["q99"]), np.array(proprio_norm_stats["q01"])
            cur_robot_state = np.where(
                mask,
                2 * (cur_robot_state - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                cur_robot_state,
            )
            cur_robot_state = np.clip(cur_robot_state, -1, 1)
            cur_robot_state = torch.tensor(cur_robot_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        
        def prepare_diffusion(input_ids_diff=None):
            # noise = torch.randn(1, self.future_action_window_size+1, action_dim, device=device)
            
            # Extract future actions
            thought_future = input_ids_diff[:, -(self.future_action_window_size+1):, :]

            # Generate noise and timesteps for diffusion
            noise = torch.randn_like(thought_future)  # [B, T, C]
            timestep = torch.randint(
                0, 
                self.diffusion.num_timesteps, 
                (thought_future.size(0),), 
                device=thought_future.device
            )
            
            # Apply diffusion sampling
            noise = self.diffusion.q_sample(thought_future, timestep, noise)

            timestep = torch.randint(0, self.diffusion.num_timesteps, (self.future_action_window_size+1,), device=device)
            using_cfg = cfg_scale > 1.0
            
            if input_ids_diff is None:
                input_ids_diff = input_ids
                if self.vlm.model_id == 'prism-dinosiglip-224px+7b':
                    input_ids_diff = input_ids_diff[:, :-2]
                elif self.vlm.model_id == 'phi-2+3b':
                    input_ids_diff = input_ids_diff[:, :-1]
            
            if using_cfg:
                noise = torch.cat([noise, noise], 0)
                uncondition = self.vlm.z_embedder.uncondition.unsqueeze(0).expand(input_ids_diff.shape[0], 1, -1)
                sample_fn = self.vlm.forward_with_cfg
                model_kwargs = {
                    'z': uncondition, 
                    'cfg_scale': cfg_scale, 
                    'input_ids': input_ids_diff, 
                }
                if cur_robot_state is not None:
                    model_kwargs['proprio'] = cur_robot_state
            else:
                model_kwargs = {'input_ids': input_ids_diff}
                if cur_robot_state is not None:
                    model_kwargs['proprio'] = cur_robot_state
                sample_fn = self.vlm.forward
            
            return noise, timestep, sample_fn, model_kwargs, using_cfg
        
        def sample_diffusion(noise, sample_fn, model_kwargs, using_cfg):
            if use_ddim and num_ddim_steps is not None:
                if self.ddim_diffusion is None:
                    self.create_ddim(ddim_step=num_ddim_steps)
                samples = self.ddim_diffusion.ddim_sample_loop(
                    sample_fn, 
                    noise.shape, 
                    noise, 
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    eta=0.0
                )
            else:
                samples = self.diffusion.p_sample_loop(
                    sample_fn, 
                    noise.shape, 
                    noise, 
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device
                )
            
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  
            
            return samples[0].cpu().numpy()
        
        def predict_diff():
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                noise, timestep, sample_fn, model_kwargs, using_cfg = prepare_diffusion()
                next_thought = sample_diffusion(noise, sample_fn, model_kwargs, using_cfg)
            return next_thought
        

        if predict_mode == 'diff':
            next_thought = predict_diff()
            return next_thought

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