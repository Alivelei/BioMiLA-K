# _*_ coding: utf-8 _*_

"""
    @Time : 2025/1/21 14:58 
    @Author : smile 笑
    @File : test2.py
    @desc :
"""


import json
from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)


# Load the model and config files
model_name = "biomedclip_local"

with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]


if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


image = torch.stack([preprocess(Image.open("CXR145_IM-0290-1001.png"))]).to(device)
image = image.unsqueeze(0) if len(image.shape) == 3 else image

b = ['adenocarcinoma histopathology', 'brain MRI', 'covid line chart', "chest X-Ray", 'no evidence of pneumonia', 'benign nodule detected']
texts = tokenizer(b, context_length=256).to(device)
print(texts)

with torch.no_grad():
    image_features, text_features, logit_scale = model(image, texts)
    print(image_features.shape)
    print(logit_scale)
    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    print(logits)
    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()
    print(sorted_indices)

