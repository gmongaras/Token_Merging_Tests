import torch
import os
from datasets import load_dataset
from diffusers.utils.testing_utils import enable_full_determinism
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np



seed = 11012197
def seed_everything(seed=42):
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)  # Set seed for Python's random module
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for CPU operations in PyTorch
    torch.cuda.manual_seed(seed)  # Set seed for GPU operations
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs if using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures that convolutions are deterministic
    torch.backends.cudnn.benchmark = False  # Slows down training but ensures reproducibility
    torch.use_deterministic_algorithms(True)
    enable_full_determinism()
seed_everything(seed)





output_path = "outputs/base"





if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load in dataset
ds = load_dataset("daspartho/stable-diffusion-prompts")["train"]

# Dataloader
batch_size = 16
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# Import model
from Models.BaseModel import get_model
pipe = get_model()
seed_everything(seed)

# Iterate through dataset
for num, batch in enumerate(loader):
    prompt = batch["prompt"]

    # Save prompts
    for i, _ in enumerate(prompt):
        j = i + num * batch_size
        with open(f"{output_path}/prompt_{j}.txt", "w", encoding="utf-8") as f:
            f.write(prompt[i].strip())

    # Load prompts back in
    for i, _ in enumerate(prompt):
        j = i + num * batch_size
        with open(f"{output_path}/prompt_{j}.txt", "r", encoding="utf-8") as f:
            prompt[i] = f.read().strip()

    # Generate image
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    images = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator
    ).images

    # Save images and prompts
    for i, image in enumerate(images):
        j = i + num * batch_size
        image.save(f"{output_path}/image_{j}.png")