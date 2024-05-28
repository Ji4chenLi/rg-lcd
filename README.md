# Reward Guided Latent Consistency Distillation

## 🔥News
- (🔥New) 05/28/2024 We release the model weights and the local gradio demo! The model weights can be download from [here](https://huggingface.co/jiachenli-ucsb/RG-LCM-SD-2.1-768-HPSv2.1). We will release other model weights soon!

- 03/18/2024 Our repo for RG-LCD is created. We will release our codes and models very soon!! Please stay tuned!

## 🏭 Installation

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 diffusers transformers accelerate gradio webdataset accelerate open_clip_torch gradio==3.48.0 
```

## ✅ Local gradio Demos (Text-to-Image):
Launch the gradio: (For MacOS users, need to set the device="mps" in app.py; For Intel GPU users, set device="xpu" in app.py)
```
python local_gradio/app.py --model_name MODEL_NAME
```
You can find the currently available models at [here](https://huggingface.co/jiachenli-ucsb) with the prefix `RG-LCM`. By default, `MODEL_NAME` is set to `jiachenli-ucsb/RG-LCM-SD-2.1-768-HPSv2.1`, which is ditilled from [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master).


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
