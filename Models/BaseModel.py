import torch
from diffusers import StableDiffusion3Pipeline

def get_model():
    # Read token from .env file
    with open(".env", "r") as f:
        token = f.read().strip()

    # Load model
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="./cache", token=token)
    pipe = pipe.to("cuda")

    return pipe