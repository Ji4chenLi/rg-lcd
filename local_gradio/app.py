# Adapted from https://github.com/luosiallen/latent-consistency-model
from __future__ import annotations

import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

import torch
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

import torch

import os
import torch
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import uuid

DESCRIPTION = """# Reward Guided Latent Consistency Models (RG-LCM)
Distilled from [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master). Check out our [Project page](https://rg-lcd.github.io) 🔥
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"


"""
   Operation System Options:
      If you are using MacOS, please set the following (device="mps") ;
      If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
      If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = torch.float16

revision = None
pretrained_teacher_model = "stabilityai/stable-diffusion-2-1"
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="vae",
    revision=revision,
    torch_dtype=torch.float16,
)
scheduler = LCMScheduler.from_pretrained(
    pretrained_teacher_model, subfolder="scheduler"
)
unet = UNet2DConditionModel.from_pretrained(
    "jiachenli-ucsb/RG-LCM-SD-2.1-768-HPSv2.1",
)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker"
)
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_teacher_model,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    revision=revision,
    torch_device=device,
    torch_dtype=DTYPE,
    safety_checker=safety_checker,
)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_image(img, profile: gr.OAuthProfile | None, metadata: dict, root_path="./"):
    unique_name = str(uuid.uuid4()) + ".png"
    unique_name = os.path.join(root_path, unique_name)
    img.save(unique_name)
    return unique_name


def save_images(image_array, profile: gr.OAuthProfile | None, metadata: dict):
    paths = []
    root_path = "./images/"
    os.makedirs(root_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_image,
                image_array,
                [profile] * len(image_array),
                [metadata] * len(image_array),
                [root_path] * len(image_array),
            )
        )
    return paths


def generate(
    prompt: str,
    seed: int = 0,
    width: int = 768,
    height: int = 768,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    param_dtype="torch.float16",
    progress=gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    pipe.to(
        torch_device=device,
        torch_dtype=torch.float16 if param_dtype == "torch.float16" else torch.float32,
    )
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        output_type="pil",
    ).images
    paths = save_images(
        result,
        profile,
        metadata={
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
    )
    print(time.time() - start_time)
    return paths, seed


examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    "A cinematic shot of robot with colorful feathers.",
    "A stunning beautiful oil painting of a lion, cinematic lighting, golden hour light.",
    "Impressionist painting of a cat, textured, hypermodern",
    "There is a secret museum of magical items inside a crystal greenhouse palace filled with intricate bookshelves, plants, and Victorian style decor",
    "a photo of an beautiful young woman wearing a floral patterned blazersitting in cafe, golden lighting, highly detailed, photo realistic.",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery", 
        )
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            randomize=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=768,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=768,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale for base",
                minimum=2,
                maximum=14,
                step=0.1,
                value=7.5,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            )
        with gr.Row():
            num_images = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=8,
                step=1,
                value=2,
                visible=True,
            )
            dtype_choices = ['torch.float16','torch.float32']
            param_dtype = gr.Radio(dtype_choices,label='torch.dtype',  
                                      value=dtype_choices[0],
                                      interactive=True,
                                      info='To save GPU memory, use torch.float16. For better quality, use torch.float32.')
    
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            num_images,
            randomize_seed,
            param_dtype,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(api_open=False)
    # demo.queue(max_size=20).launch()
    demo.launch()
