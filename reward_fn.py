from typing import List
import os
from os.path import expanduser
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import open_clip_customized.src.open_clip as open_clip
from transformers import AutoModel, AutoProcessor
from torchvision.transforms import Normalize, Resize, InterpolationMode

# Image processing
CLIP_RESIZE = Resize(224, interpolation=InterpolationMode.BICUBIC)
CLIP_NORMALIZE = Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


# from https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
def get_aesthetic_model(clip_model="vit_l_14"):
    """
    Get an aesthetic scoring model based off of clip vit_l_14 or clip vit_b_32

    """
    # Download to cache folder
    # Aesthetic model is simple linear layer on top of CLIP stem
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


def aesthetic_score(image, model, aesthetic_model):
    """
    Get aesthetic score of image (possibly stack of images from multicrop)
    Inputs:
    * image (bs, 3, 224, 224) tensor
    * model: clip feature extractor
    * aesthetic_model: linear head

    Output:
    * Single scalar score
    """
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    prediction = aesthetic_model(image_features)
    return prediction


def get_aesthetic_score_fn(
    precision="fp32",
    clip_model_str="vit_l_14",
    weights=[1, 1],
):
    """
    Loss function for aesthetics

    Inputs:
    * aesthetic value to target in 1-10. If None will maximize aesthetic vlaue
    * clip_model_str: vit_l_14 or vit_b_32 or 'both' , which aesthetic model to use
    * weights (list of floats): Weights of vit_b_32 vs vit_l_14 if using 'both'
    """
    # https://github.com/LAION-AI/aesthetic-predictor


    # Create normal clip model stems
    if clip_model_str == "both":
        model_l, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        amodel_l = get_aesthetic_model(clip_model="vit_l_14")
        amodel_l.eval()
        model_b, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        amodel_b = get_aesthetic_model(clip_model="vit_b_32")
        amodel_b.eval()
        models = [model_l, model_b]
        amodels = [amodel_l, amodel_b]
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14" if clip_model_str == "vit_l_14" else "ViT-B-32",
            pretrained="openai",
        )
        amodel = get_aesthetic_model(clip_model=clip_model_str)
        amodel.eval()
        models = [model]
        amodels = [amodel]

    assert precision in ["fp32", "fp16"]
    precision = torch.float32 if precision == "fp32" else torch.float16
    for m in models + amodels:
        m.requires_grad_(False)
        m.to(precision)

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: str, return_logits=False):
        device = image_inputs.device
        del text_inputs, return_logits
        # Process pixels and multicrop
        x_var = CLIP_RESIZE(image_inputs)
        x_var = CLIP_NORMALIZE(x_var)

        for model, amodel in zip(models, amodels):
            model.to(device)
            amodel.to(device)

        # Get predicted scores from model(s)
        predictions = [
            aesthetic_score(x_var, model, amodel)
            for model, amodel in zip(models, amodels)
        ]
        # Average predictions across models
        score = sum([w * p for w, p in zip(weights, predictions)]) / len(
            predictions
        )

        return score

    return score_fn


def get_pick_score_fn(precision="fp32"):
    """
    Loss function for PICK SCORE
    """
    print("Loading PICK SCORE model")

    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval()
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.requires_grad_(False)
    if precision == "fp16":
        model.to(torch.float16)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str, return_logits=False):
        device = image_inputs.device
        model.to(device)

        pixel_values = CLIP_RESIZE(image_inputs)
        pixel_values = CLIP_NORMALIZE(CLIP_RESIZE(image_inputs))

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.no_grad():
            preprocessed = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_embs = model.get_text_features(**preprocessed)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Get predicted scores from model(s)
        score = (text_embs * image_embs).sum(-1)
        if return_logits:
            score = score * model.logit_scale.exp()
        return score

    return score_fn


def get_hpsv2_fn(precision="amp"):
    precision = "amp" if precision == "no" else precision
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        f"{os.environ['HOME']}/.cache/hpsv2/HPS_v2.1_compressed.pt",
        # f"{os.environ['HOME']}/.cache/hpsv2/HPS_v2_compressed.pt",
        precision=precision,
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    tokenizer = get_tokenizer("ViT-H-14")
    model.eval()
    model.requires_grad_(False)

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        for t in preprocess_val.transforms[2:]:
            image_inputs = torch.stack([t(img) for img in image_inputs])

        if isinstance(text_inputs[0], str):
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        hps_score = (image_features * text_features).sum(-1)
        if return_logits:
            hps_score = hps_score * model.logit_scale.exp()
        return hps_score

    return score_fn


def get_img_reward_fn(precision="fp32"):
    # pip install image-reward
    import ImageReward as RM
    import torch.nn.functional as F
    from torchvision.transforms import Compose, Resize, CenterCrop
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC

    model = RM.load("ImageReward-v1.0")
    model.eval()
    model.requires_grad_(False)

    rm_preprocess =  Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        CLIP_NORMALIZE,
    ])

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        del return_logits
        device = image_inputs.device
        model.to(device)
        if precision == "fp16":
            model.to(torch.float16)
        elif precision == "bf16":
            model.to(torch.bfloat16)

        image = rm_preprocess(image_inputs).to(device)
        text_input = model.blip.tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        rewards = model.score_gard(text_input.input_ids, text_input.attention_mask, image)
        return -F.relu(-rewards+2).squeeze(-1)

    return score_fn


def get_clip_score_fn(precision="amp"):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision=precision,
        device="cuda",
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
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model.eval()
    model.requires_grad_(False)

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        if isinstance(text_inputs[0], str):
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        clip_score = (image_features * text_features).sum(-1)
        if return_logits:
            clip_score = clip_score * model.logit_scale.exp()
        return clip_score

    return score_fn


def get_weighted_hpsv2_img_reward_fn(precision="amp", weights=[1., 0.1]):
    hpsv2_score_fn = get_hpsv2_fn(precision)
    img_reward_score_fn = get_img_reward_fn(precision)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        hpsv2_score = hpsv2_score_fn(image_inputs, text_inputs)
        img_reward_score = img_reward_score_fn(image_inputs, text_inputs)
        return weights[0] * hpsv2_score + weights[1] * img_reward_score
    return score_fn

def get_latent_clip_score_fn(precision="amp"):
    model, _, _ = open_clip.create_model_and_transforms(
        "latent_RN50",
        "laion2B-s32B-b79K",
        precision=precision,
        device="cuda",
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
    checkpoint_path = "/home/jiachenli/open_clip/logs/2024_02_11-17_05_15-model_latent_RN50-lr_0.0005-b_480-j_8-p_amp/checkpoints/epoch_32.pt"
    state_dict = open_clip.factory.load_process_state_dict(model, checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def score_fn(latents: torch.Tensor, text_inputs: str):
        # Process pixels and multicrop
        device = latents.device
        model.to(device)
        # embed
        image_features = model.encode_image(latents, normalize=True)
        with torch.no_grad():
            text_inputs = tokenizer(text_inputs).to(device)
            text_features = model.encode_text(text_inputs, normalize=True)
        score = (image_features * text_features).sum(-1)
        return score

    return score_fn

def get_reward_fn(reward_fn_name: str, **kwargs):
    if reward_fn_name == "aesthetic":
        return get_aesthetic_score_fn(**kwargs)
    elif reward_fn_name == "pick":
        return get_pick_score_fn(**kwargs)
    elif reward_fn_name == "hpsv2":
        return get_hpsv2_fn(**kwargs)
    elif reward_fn_name == "img_reward":
        return get_img_reward_fn(**kwargs)
    elif reward_fn_name == "clip":
        return get_clip_score_fn(**kwargs)
    elif reward_fn_name == "latent_clip":
        return get_latent_clip_score_fn(**kwargs)
    elif reward_fn_name == "weighted_hpsv2_img_reward":
        return get_weighted_hpsv2_img_reward_fn(**kwargs)
    else:
        raise ValueError("Invalid reward_fn_name")
