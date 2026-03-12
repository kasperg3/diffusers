import re
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
from vae import AutoencoderKL
from unet import UNet2DConditionModel
from tqdm import tqdm

from fliser import Fliser

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class S3Diff(torch.nn.Module):
    def __init__(
        self,
        base_model="stabilityai/sd-turbo",
        vae_lora_path=None,
        unet_lora_path=None,
        rest_path=None,
        
        lora_rank_unet=32,
        lora_rank_vae=16,
        block_embedding_dim=64,
        latent_tiled_size=96,
        latent_tiled_overlap=32,
        device= torch.device("cpu"),
    ):
        super().__init__()
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        self.sched = make_1step_sched(base_model, device=device)
        self.guidance_scale = 1.07

        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", state_dict_path=vae_lora_path)
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", state_dict_path=unet_lora_path)

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )


        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        self._load_rest_state_dict(rest_path)
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999]).long()
        self.text_encoder.requires_grad_(False)
        self.device = device
        self.to(device)

        # vae tile
        self.vae.enable_tiling()
    
    def _load_rest_state_dict(self, state_dict_path: str):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        
        self.vae_de_mlp.load_state_dict(state_dict["state_dict_vae_de_mlp"]) 
        self.unet_de_mlp.load_state_dict(state_dict["state_dict_unet_de_mlp"])
        self.vae_block_mlp.load_state_dict(state_dict["state_dict_vae_block_mlp"])
        self.unet_block_mlp.load_state_dict(state_dict["state_dict_unet_block_mlp"])
        self.vae_fuse_mlp.load_state_dict(state_dict["state_dict_vae_fuse_mlp"])
        self.unet_fuse_mlp.load_state_dict(state_dict["state_dict_unet_fuse_mlp"])
        state_embeddings = state_dict["state_embeddings"]
        self.unet_block_embeddings.load_state_dict(state_embeddings["state_dict_unet_block"])
        self.vae_block_embeddings.load_state_dict(state_embeddings["state_dict_vae_block"])
        self.W = nn.Parameter(state_dict["w"], requires_grad=False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
    
    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        self.vae_de_mlp.to(*args, **kwargs)
        self.unet_de_mlp.to(*args, **kwargs)
        self.vae_block_mlp.to(*args, **kwargs)
        self.unet_block_mlp.to(*args, **kwargs)
        self.vae_fuse_mlp.to(*args, **kwargs)
        self.unet_fuse_mlp.to(*args, **kwargs)
        self.vae_block_embeddings.to(*args, **kwargs)
        self.unet_block_embeddings.to(*args, **kwargs)
        self.timesteps = self.timesteps.to(*args, **kwargs).long()
        return super().to(*args, **kwargs)
    
    def _encode_prompts(self, pos_prompt, neg_prompt, device):
        # encode the text prompt
        pos_caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length,
                                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        pos_caption_enc = self.text_encoder(pos_caption_tokens)[0]

        # encode the text prompt
        neg_caption_tokens = self.tokenizer(neg_prompt, max_length=self.tokenizer.model_max_length,
                                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        neg_caption_enc = self.text_encoder(neg_caption_tokens)[0]

        return pos_caption_enc, neg_caption_enc
    
    def _compute_embeds(self, deg_score):
        # degradation fourier embedding
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))
        
        return vae_embeds, unet_embeds
    
    def _patch_using_embeds(self, vae_embeds, unet_embeds):
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")
                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)
    
    @torch.no_grad()
    def forward(self, c_t, deg_score, pos_prompt, neg_prompt):
        device = self.device

        # Pre-processing
        pos_caption_enc, neg_caption_enc = self._encode_prompts(pos_prompt, neg_prompt, device)
        vae_embeds, unet_embeds = self._compute_embeds(deg_score)
        self._patch_using_embeds(vae_embeds, unet_embeds) 

        # VAE encode
        lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        ## add tile function
        _, c, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
            neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
            model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            # print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")

            # Instantiate tiler
            tile_size = min(tile_size, min(h, w))
            fliser = Fliser(
                image_size=(h, w),
                num_channels=c,
                tile_size=(tile_size, tile_size),
                min_overlap=tile_overlap,
                device=lq_latent.device,
            )

            for tile in tqdm(fliser.tiles(), desc="Processing Tiles using UNet", leave=False):
                pos_model_pred = self.unet(
                    lq_latent[tile.slice()],
                    self.timesteps,
                    encoder_hidden_states=pos_caption_enc
                ).sample
                neg_model_pred = self.unet(
                    lq_latent[tile.slice()],
                    self.timesteps,
                    encoder_hidden_states=neg_caption_enc
                ).sample
                pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
                fliser.update(tile, pred) 
            
            model_pred = fliser.compute()
            
        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image