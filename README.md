# Reward Guided Latent Consistency Distillation

## 🔥News
- (🔥New) 03/18/2024 Our repo for RG-LCD is created. We will release our codes and models very soon!! Please stay tuned!

## 🏭 Installation

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 diffusers transformers accelerate gradio webdataset accelerate open_clip_torch
```

## 🏋️ Training commands

To perform RG-LCD with the HPSv2.1, we can run

```
accelerate launch main.py --output_dir=PATH_TO_LOG \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention --resolution 768 \
 --train_shards_path_or_url "pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01090}.tar?download=true" \
 --optimize_reward_fn \
 --reward_fn_name hpsv2 \
 --direct_optim_expert_reward 
 --reward_scale 1
```

## 📃 Citation

```bibtex
@misc{li2024reward,
    title={Reward Guided Latent Consistency Distillation},
    author={Jiachen Li and Weixi Feng and Wenhu Chen and William Yang Wang},
    eprint={2403.11027},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
