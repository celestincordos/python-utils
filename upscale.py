import argparse
from pathlib import Path
from PIL import Image
import requests
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np 

model_path =Path ("./resources/models/RealESRGAN_x4plus.pth")

def upscale_with_realesrgan(input_image: Image.Image, scale: int = 2) -> Image.Image:
    torch.cuda.empty_cache()

    # state_dict = torch.load(model_path, map_location=torch.device('gpu'))
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer( scale=scale, model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", model=model, tile=0, pre_pad=0, half=0, device='cpu')
    
    
    image = input_image.convert("RGB")
    img = np.array(image)
    output =  upsampler.enhance(img)
    return Image.fromarray(output)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the background from an image using rembg."
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument(
        "-o", "--output", help="Path to save the output image (optional)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Open input image
    input_image = Image.open(input_path)

    # Process background removal
    output_image = upscale_with_realesrgan(input_image, scale=2)

    # Determine output path
    output_path = args.output or input_path.with_name(f"{input_path.stem}_no_bg.png")

    # Save result
    output_image.save(output_path)
    print(f"✅ Background removed: {output_path}")


if __name__ == "__main__":
    main()