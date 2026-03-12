from math import ceil, floor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import interpolate, pad
from torchvision.transforms.functional import to_pil_image, to_tensor

from de_net import DEResNet
from s3diff_tile import S3Diff
from wavelet_color import adain_color_fix, wavelet_color_fix


class Pipeline:
    
    def __init__(
        self,
        net_sr: S3Diff,
        net_de: DEResNet,
        scale_factor: int = 4,
        device: torch.device = torch.device("cpu"),
        align_method: str ="wavelet",
        pos_prompt: str = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt: str = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"
    ):
        self.net_sr = net_sr
        self.net_de = net_de
        self.scale_factor = scale_factor
        self.device = device
        self.align_method = align_method
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt
   
    def _get_size(self, image1, image2):
        size = image1.size
        if image1.size != image2.size:
            size = (min(image1.width, image2.width), min(image1.height, image2.height))
        if size[0] % 64 != 0 or size[1] % 64 != 0:
            size = (floor(size[0] / 64) * 64, floor(size[1] / 64) * 64)
        return size

    def _infer(
        self,
        image: Image.Image
    ):
        # Degradation score
        image_tensor = to_tensor(image).unsqueeze(0).to(self.device)
        deg_score = self.net_de(image_tensor)
        
        # Normalize
        image_norm = image_tensor * 2 - 1.0
        image_norm = image_norm.clamp(-1.0, 1.0)
        
        # Super-resolution
        output = self.net_sr(
            image_norm,
            deg_score,
            pos_prompt=[self.pos_prompt],
            neg_prompt=[self.neg_prompt],
        ).cpu().detach()
        output = (output + 1.0) / 2.0
        output = to_pil_image(output.squeeze(0))
        
        # Color alignment
        if self.align_method == "wavelet":
            output = wavelet_color_fix(output, image)
        elif self.align_method == "adain":
            output = adain_color_fix(output, image)
        return output
        

    @torch.inference_mode()
    def generate_training_sample(
        self,
        image_empty: Image.Image,
        image_furnished: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:

        # Align image sizes
        size_hr = self._get_size(image_empty, image_furnished)
        size_lr = (size_hr[0] // self.scale_factor, size_hr[1] // self.scale_factor)
        image_empty_aligned = image_empty.resize(size_hr, Image.Resampling.BILINEAR) if image_empty.size != size_hr else image_empty
        image_furnished_aligned = image_furnished.resize(size_hr, Image.Resampling.BILINEAR) if image_furnished.size != size_hr else image_furnished

        # Degrade images
        image_empty_degraded = image_empty.resize(size_lr, Image.Resampling.BILINEAR).resize(size_hr, Image.Resampling.BILINEAR)
        image_furnished_degraded = image_furnished.resize(size_lr, Image.Resampling.BILINEAR).resize(size_hr, Image.Resampling.BILINEAR)

        # Super-resolution
        image_empty_sr = self._infer(image_empty_degraded)
        image_furnished_sr = self._infer(image_furnished_degraded)

        return image_empty_aligned, image_furnished_aligned, image_empty_sr, image_furnished_sr

def get_models(
    checkpoint_dir: str,
    device,
    dtype
) -> tuple[S3Diff, DEResNet]:
    net_sr = S3Diff(
        base_model="stabilityai/sd-turbo",
        vae_lora_path=f"{checkpoint_dir}/vae.pth",
        unet_lora_path=f"{checkpoint_dir}/unet.pth",
        rest_path=f"{checkpoint_dir}/rest.pth",
        latent_tiled_size=96,
        latent_tiled_overlap=32,
        device=device,
        )
    net_sr.set_eval()
    net_sr.to(dtype=dtype)

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(f"{checkpoint_dir}/de_net.pth")
    net_de.eval()
    net_de.to(device, dtype=dtype)
    return net_sr, net_de

def get_csv(path: str) -> tuple[pd.DataFrame, Path]:
    root = Path(path).parent
    df = pd.read_csv(path).reset_index(drop=True)
    df['unfurnished'] = df['unfurnished'].apply(lambda x: str(root / x))
    df['furnished'] = df['furnished'].apply(lambda x: str(root / x))
    df['mask'] = df['mask'].apply(lambda x: str(root / x))
    return df, root

def main():
    csv = "/mnt/nvme2/datasets/VS-1.0/dataframe.csv"
    device = torch.device("cuda:1")
    dtype = torch.float32
    scale_factor = 4
    align_method = "wavelet"

    df, root = get_csv(csv) 
    save_dir = root / 's3diff'
    save_dir.mkdir(parents=True, exist_ok=True)

    net_sr, net_de = get_models(device, dtype)
    pipeline = Pipeline(
        net_sr=net_sr,
        net_de=net_de,
        scale_factor=scale_factor,
        device=device,
        align_method=align_method,
    )
    
    new_df = [] 
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}")
        try: 
            image_empty = Image.open(row['unfurnished']).convert('RGB')
            image_furnished = Image.open(row['furnished']).convert('RGB')

            image_empty_out, image_furnished_out, image_empty_sr, image_furnished_sr = pipeline.generate_training_sample(
                image_empty=image_empty,
                image_furnished=image_furnished,
            )

            # As the images may have been resized, we need to resize the masks accordingly
            mask = Image.open(row['mask']).convert('L')
            mask_resized = mask.resize(image_empty_out.size, Image.Resampling.BILINEAR)

            # Save
            empty_name = Path(row['unfurnished']).stem
            furnished_name = Path(row['furnished']).stem
            mask_name = Path(row['mask']).stem

            empty_gt_path = save_dir / f"{empty_name}_gt.jpg"
            furnished_gt_path = save_dir / f"{furnished_name}_gt.jpg"
            empty_sr_path = save_dir / f"{empty_name}_sr.jpg"
            furnished_sr_path = save_dir / f"{furnished_name}_sr.jpg"
            mask_path = save_dir / f"{mask_name}_mask.png"

            image_empty_out.save(empty_gt_path, quality=95)
            image_furnished_out.save(furnished_gt_path, quality=95)
            image_empty_sr.save(empty_sr_path, quality=95)
            image_furnished_sr.save(furnished_sr_path, quality=95)
            mask_resized.save(mask_path, quality=95)

            new_df.append({
                'unfurnished_gt': str(empty_gt_path.relative_to(root)),
                'furnished_gt': str(furnished_gt_path.relative_to(root)),
                'unfurnished_sr': str(empty_sr_path.relative_to(root)),
                'furnished_sr': str(furnished_sr_path.relative_to(root)),
                'mask': str(mask_path.relative_to(root)),
            })
            
            if idx % 5 == 0:
                pd.DataFrame(new_df).to_csv(root / 's3diff.csv', index=False)
        except Exception as e:
            print(f"Error processing {row['unfurnished']} and {row['furnished']}: {e}")
            continue

    pd.DataFrame(new_df).to_csv(root / 's3diff.csv', index=False)

if __name__ == "__main__":
    main()