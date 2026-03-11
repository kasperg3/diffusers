from diffusers import AutoencoderKL
from peft import LoraConfig
import torch

class AutoencoderKL(AutoencoderKL):
   
    @classmethod 
    def from_pretrained(cls, *args, state_dict_path=None, **kwargs):
        self = super().from_pretrained(*args, **kwargs)
        self.add_adapter(
            LoraConfig(
                r=16,
                init_lora_weights="gaussian",
                target_modules='^encoder\\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\\.0)$'
            ),
            adapter_name="vae_skip"
        )

        if state_dict_path is not None:
            state_dict_vae = torch.load(state_dict_path)
            self.load_state_dict(state_dict_vae)
        
        return self