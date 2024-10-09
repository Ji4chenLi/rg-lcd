#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version

from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import open_clip_customized.src.open_clip as open_clip
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from reward_fn import get_reward_fn
from ddim_solver import DDIMSolver
from dataset import SDText2ImageDataset
from common_utils import (
    update_ema,
    predicted_noise,
    predicted_origin,
    guidance_scale_embedding,
    append_dims,
    scalings_for_boundary_conditions,
)


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


def log_validation(vae, unet, args, accelerator, weight_dtype, step, name="target"):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        unet=unet,
        scheduler=LCMScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        ),
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "shorn in the marrow of its fossil",
        "Portrait of a female elf warlock, long pointy ears, glowing green eyes, bushy red hair and freckles",
        "Mountains range with waterfall, purple haze, art by greg rutkowski and magali villeneuve, artstation.",
        "A half - masked rugged laboratory engineer man with cybernetic enhancements as seen from a distance, scifi character portrait by greg rutkowski, esuthio, craig mullins.",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        with torch.autocast("cuda"):
            images = pipeline(
                prompt=prompt,
                num_inference_steps=4,
                num_images_per_prompt=4,
                generator=generator,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({f"validation/{name}": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--latent_rm_type",
        type=str,
        default="latent_ViT",
        help="Path to pretrained Latent Reward Model.",
    )
    parser.add_argument(
        "--unlocked_rm_text_layers",
        type=int,
        default=5,
        help="Path to pretrained Latent Reward Model.",
    )

    parser.add_argument(
        "--pretrained_latent_rm",
        type=str,
        default=None,
        help="Path to pretrained Latent Reward Model.",
    )

    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=453645634, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01090}.tar?download=true",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--rm_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=10900000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--rm_lr",
        type=float,
        default=0.0000033,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="huber",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=32,
        required=False,
        help=(
            "The batch size used when encoding images to latents using the VAE."
            " Encoding the whole batch at once may run into OOM issues."
        ),
    )
    parser.add_argument(
        "--vae_decode_batch_size",
        type=int,
        default=16,
        required=False,
        help=(
            "The batch size used when decoding images to latents using the VAE."
            " Decoding the whole batch at once may run into OOM issues."
        ),
    )

    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--optimize_reward_fn",
        action="store_true",
        default=False,
        help="Whether to optimize the reward function",
    )
    parser.add_argument(
        "--guide_with_expert_reward",
        action="store_true",
        default=False,
        help="Whether to optimize the reward function",
    )
    parser.add_argument(
        "--reward_fn_name",
        type=str,
        default="hpsv2",
        help="Whether to optimize the reward function",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--rm_loss_scale",
        type=float,
        default=1.0,
        help="The scale of the loss for training the reward model",
    )
    parser.add_argument(
        "--latent_rm_loss_type",
        type=str,
        default="distillation",
        choices=["cross_entropy", "distillation"],
        help="",
    )
    parser.add_argument(
        "--smooth_gt_label",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--direct_optim_expert_reward",
        action="store_true",
        default=False,
        help="",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


def huber_loss(pred, target, huber_c=0.001):
    loss = torch.sqrt((pred.float() - target.float()) ** 2 + huber_c**2) - huber_c
    return loss.mean()


def main(args):
    try:
        args.work_on_latent = (
            True
            if os.path.exists("/".join(args.train_shards_path_or_url.split("/")[:-1]))
            else False
        )
    except Exception:
        args.work_on_latent = False

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    if args.optimize_reward_fn:
        reward_fn = get_reward_fn(args.reward_fn_name, precision=args.mixed_precision)

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="scheduler",
        revision=args.teacher_revision,
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SD 1.X/2.X checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    # 3. Load text encoders from SD 1.X/2.X checkpoint.
    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )

    # 4. Load VAE from SD 1.X/2.X checkpoint
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD 1.X/2.X checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # 7. Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = (
        teacher_unet.config.time_cond_proj_dim
        if teacher_unet.config.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    unet = UNet2DConditionModel.from_config(
        teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
    )
    # load teacher_unet weights into unet
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()

    if not args.direct_optim_expert_reward and args.optimize_reward_fn:
        latent_rm, _, _ = open_clip.create_model_and_transforms(
            args.latent_rm_type,
            args.pretrained_latent_rm,
            precision="amp",
            device=accelerator.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            image_mean=None,
            image_std=None,
            image_interpolation=None,
            image_resize_mode=None,  # only effective for inference
            aug_cfg={},
            pretrained_image=False,
            output_dict=True,
        )
        rm_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        del latent_rm.vae

        # Freeze certain amount of layers of Latent RM
        if args.guide_with_expert_reward:
            if args.latent_rm_type == "latent_ViT":
                latent_rm.lock_image_tower(unlocked_groups=2)
            else:
                latent_rm.visual.requires_grad_(True)
                for n, p in latent_rm.visual.named_parameters():
                    p.requires_grad_(False)
                    if "attnpool" in n:
                        break
            latent_rm.lock_text_tower(unlocked_layers=args.unlocked_rm_text_layers)

        def latent_score_fn(
            rm, latents: torch.Tensor, text_inputs: List[str], return_logits=False
        ):
            # Process pixels and multicrop
            device = latents.device
            if isinstance(text_inputs[0], str):
                text_inputs = rm_tokenizer(text).to(device)

            output = rm(latents, text_inputs)
            score = (output["image_features"] * output["text_features"]).sum(-1)
            if return_logits:
                score = score * output["logit_scale"]
            return score

    # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
    # Initialize from (online) unet
    target_unet = UNet2DConditionModel.from_config(
        teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
    )
    target_unet.load_state_dict(unet.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                target_unet.save_pretrained(os.path.join(output_dir, "unet_target"))

                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(
                os.path.join(input_dir, "unet_target")
            )
            target_unet.load_state_dict(load_model.state_dict())
            target_unet.to(accelerator.device)
            del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            target_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.guide_with_expert_reward and not args.direct_optim_expert_reward:
        rm_optimizer = optimizer_class(
            latent_rm.parameters(),
            lr=args.rm_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return {"prompt_embeds": prompt_embeds}

    dataset = SDText2ImageDataset(
        train_shards_path_or_url=args.train_shards_path_or_url,
        num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
        resolution=args.resolution,
        interpolation_type=args.interpolation_type,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        work_on_latent=args.work_on_latent,
    )
    train_dataloader = dataset.train_dataloader

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    if args.guide_with_expert_reward and not args.direct_optim_expert_reward:
        latent_rm, rm_optimizer = accelerator.prepare(latent_rm, rm_optimizer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}},
        )

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # 16. Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Load and process the image and text conditioning
                if args.work_on_latent:
                    with torch.no_grad():
                        params, text = batch
                        params = params.to(accelerator.device, non_blocking=True)
                        latents = DiagonalGaussianDistribution(params).sample()
                else:
                    image, text = batch
                    image = image.to(accelerator.device, non_blocking=True)
                    pixel_values = image.to(dtype=weight_dtype)
                    # encode pixel values with batch size of at most args.vae_encode_batch_size
                    latents = []
                    for i in range(
                        0, pixel_values.shape[0], args.vae_encode_batch_size
                    ):
                        latents.append(
                            vae.encode(
                                pixel_values[i : i + args.vae_encode_batch_size]
                            ).latent_dist.sample()
                        )
                    latents = torch.cat(latents, dim=0)

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                encoded_text = compute_embeddings_fn(text)
                bsz = latents.shape[0]

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = (
                    noise_scheduler.config.num_train_timesteps
                    // args.num_ddim_timesteps
                )
                index = torch.randint(
                    0, args.num_ddim_timesteps, (bsz,), device=latents.device
                ).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [
                    append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(
                    latents, noise, start_timesteps
                )

                # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w_embedding = guidance_scale_embedding(
                    w, embedding_dim=time_cond_proj_dim
                )
                w = w.reshape(bsz, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # 6. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample

                pred_x_0 = predicted_origin(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        cond_pred_x0 = predicted_origin(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = predicted_origin(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (
                            cond_pred_noise - uncond_pred_noise
                        )
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_noise_pred = target_unet(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                    pred_x_0 = predicted_origin(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # 10. Calculate loss
                if args.loss_type == "l2":
                    distill_loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                elif args.loss_type == "huber":
                    distill_loss = huber_loss(model_pred, target, args.huber_c)

                if args.optimize_reward_fn:
                    pred_latents = model_pred / vae.config.scaling_factor
                    if args.direct_optim_expert_reward:
                        decoded_imgs = vae.decode(
                            pred_latents.to(vae.dtype),
                            return_dict=False,
                        )[0]
                        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                        expert_rewards = reward_fn(decoded_imgs, text)
                        reward_loss = -expert_rewards.mean() * args.reward_scale
                    else:
                        latent_rm.requires_grad_(False)
                        latent_rm.eval()

                        text_inputs = rm_tokenizer(text).to(accelerator.device)
                        latent_rewards = latent_score_fn(
                            latent_rm, pred_latents, text_inputs
                        )
                        reward_loss = -latent_rewards.mean() * args.reward_scale
                    loss = distill_loss + reward_loss
                else:
                    loss = distill_loss

                accelerator.backward(loss)
                # 11. Backpropagate on the online student model (`unet`)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if args.optimize_reward_fn and args.guide_with_expert_reward:
                    latent_rm.requires_grad_(True)
                    latent_rm.train()

                    indices = torch.randperm(args.train_batch_size)[
                        : args.rm_batch_size
                    ]
                    pred_latents = pred_latents[indices]
                    target_latents = target[indices] / vae.config.scaling_factor
                    target_latents = target_latents.to(pred_latents.dtype)
                    gt_latents = latents[indices] / vae.config.scaling_factor
                    gt_latents = gt_latents.to(pred_latents.dtype)
                    text_inputs = text_inputs[indices]

                    concat_latents = torch.cat(
                        [pred_latents, target_latents, gt_latents], dim=0
                    )
                    concat_text = torch.cat([text_inputs, text_inputs, text_inputs])
                    expert_return_logits = args.latent_rm_loss_type == "distillation"
                    with torch.no_grad():
                        decoded_imgs = []
                        for i in range(
                            0, concat_latents.shape[0], args.vae_decode_batch_size
                        ):
                            imgs = vae.decode(
                                concat_latents[i : i + args.vae_decode_batch_size].to(
                                    vae.dtype
                                ),
                                return_dict=False,
                            )[0]
                            decoded_imgs.append(imgs)

                        decoded_imgs = (torch.cat(decoded_imgs) / 2 + 0.5).clamp(0, 1)
                        if args.reward_fn_name != "hpsv2":
                            expert_rewards = reward_fn(
                                decoded_imgs, text + text + text, expert_return_logits
                            )
                            if (
                                expert_return_logits
                                and args.reward_fn_name == "img_reward"
                            ):
                                expert_rewards *= latent_rm.module.logit_scale.exp()
                        else:
                            expert_rewards = reward_fn(
                                decoded_imgs, concat_text, expert_return_logits
                            )
                        expert_rewards = torch.stack(torch.chunk(expert_rewards, 3))

                        # imgs = vae.decode(concat_latents, return_dict=False)[0]
                        # imgs = (imgs / 2 + 0.5).clamp(0, 1)
                        # expert_rewards = reward_fn(imgs, (text + text))

                    pred_logits = latent_score_fn(
                        latent_rm,
                        concat_latents.detach(),
                        concat_text,
                        return_logits=True,
                    )
                    latent_logits = torch.stack(torch.chunk(pred_logits, 3))

                    rm_loss = 0.0
                    rm_accuracy = 0.0
                    for i in range(2):
                        for j in range(i + 1, 3):
                            logits = torch.stack(
                                [latent_logits[i], latent_logits[j]]
                            ).transpose(0, 1)
                            expert_logits = torch.stack(
                                [expert_rewards[i], expert_rewards[j]]
                            ).transpose(0, 1)
                            labels = expert_logits.argmax(dim=1)
                            if args.latent_rm_loss_type == "distillation":
                                # expert_rewards is the logits of the expert distribution
                                expert_prob = F.softmax(expert_logits, dim=1)
                                if j == 2 and args.smooth_gt_label:
                                    for k, a in enumerate(labels):
                                        if a == 0:
                                            expert_prob[k] = (
                                                torch.ones_like(expert_prob[k]) / 2
                                            )
                                rm_loss += F.kl_div(
                                    F.log_softmax(logits, dim=1),
                                    expert_prob,
                                    reduction="batchmean",
                                )
                            elif args.latent_rm_loss_type == "cross_entropy":
                                labels = expert_rewards.argmax(dim=1)
                                if j == 2:
                                    # Ground truth images should be always the best
                                    labels = torch.ones_like(labels)
                                rm_loss += F.cross_entropy(logits, labels)
                            with torch.no_grad():
                                rm_accuracy += (
                                    (logits.argmax(dim=1) == labels).float().mean()
                                )

                    rm_accuracy /= 3

                    accelerator.backward(rm_loss * args.rm_loss_scale)
                    rm_optimizer.step()
                    rm_optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 12. Make EMA update to target student model parameters (`target_unet`)
                update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d
                                for d in checkpoints
                                if d.startswith("checkpoint") and not "rm" in d
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        if args.guide_with_expert_reward:
                            latent_rm_state_dict = accelerator.unwrap_model(
                                latent_rm
                            ).state_dict()
                            rm_ckpt_dict = {
                                "global_step": global_step,
                                "optimizer": optimizer.state_dict(),
                                "latent_rm": latent_rm_state_dict,
                            }
                            torch.save(rm_ckpt_dict, save_path + "-rm.pt")

                    if global_step % args.validation_steps == 0:
                        torch.cuda.empty_cache()
                        log_validation(
                            vae,
                            target_unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "target",
                        )
                        log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "online",
                        )

            logs = {
                "distillation loss": distill_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if args.optimize_reward_fn:
                logs["Reward Loss"] = reward_loss.detach().item()
                if args.direct_optim_expert_reward:
                    logs[f"{args.reward_fn_name} rewards"] = (
                        expert_rewards.mean().detach().item()
                    )
                else:
                    logs["Latent rewards"] = latent_rewards.mean().detach().item()
                if args.guide_with_expert_reward:
                    logs[f"Expert {args.reward_fn_name} rewards"] = (
                        expert_rewards.mean().detach().item()
                    )
                    logs["Reward Acc"] = rm_accuracy.detach().item()
                    logs["RM Loss"] = rm_loss.detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        target_unet = accelerator.unwrap_model(target_unet)
        target_unet.save_pretrained(os.path.join(args.output_dir, "unet_target"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
