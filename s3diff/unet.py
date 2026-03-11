from diffusers import UNet2DConditionModel
from peft import LoraConfig
import torch

class UNet2DConditionModel(UNet2DConditionModel):
   
    @classmethod 
    def from_pretrained(cls, *args, state_dict_path=None, **kwargs):
        self = super().from_pretrained(*args, **kwargs)
        self.add_adapter(
            LoraConfig(
                r=32,
                init_lora_weights="gaussian",
                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0', 'conv', 'conv1', 'conv2', 'conv_shortcut', 'conv_out', 'proj_in', 'proj_out', 'ff.net.2', 'ff.net.0.proj']
            ),
        )

        if state_dict_path is not None:
            state_dict_vae = torch.load(state_dict_path)
            self.load_state_dict(state_dict_vae)
        
        return self



