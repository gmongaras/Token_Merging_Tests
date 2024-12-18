import torch
import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from diffusers.utils.testing_utils import enable_full_determinism
import random
import numpy as np





output_path = "outputs/KeyValueMerge100_Mean"
class_name = "KeyValueMerge100_Mean"







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


class PromptDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.prompt_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".txt")]

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        prompt_path = os.path.join(self.folder_path, f"prompt_{idx}.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        # What is the name of the file?
        # img_num = self.prompt_files[idx].split(".")[0].split("_")[-1]
        img_num = idx

        return {"prompt": prompt, "img_num": img_num}





if not os.path.exists(output_path):
    os.makedirs(output_path)

# Dataset
ds = PromptDataset("outputs/base")

# Dataloader
batch_size = 16
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

# Import model based on the class name
module = __import__(f"Models.{class_name}", fromlist=[class_name])
pipe = module.get_model()
seed_everything(seed)

# Iterate through dataset
for num, prompts in enumerate(loader):
    image_nums = prompts["img_num"]
    prompts = prompts["prompt"]

    # Generate image
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    images = pipe(
        prompts,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator
    ).images

    # Save images and prompts
    for i, image in enumerate(images):
        image.save(f"{output_path}/image_{image_nums[i]}.png")
        with open(f"{output_path}/prompt_{image_nums[i]}.txt", "w", encoding="utf-8") as f:
            f.write(prompts[i].strip())