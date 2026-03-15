# _*_ coding: utf-8 _*_

"""
    @Time : 2025/1/21 14:35 
    @Author : smile 笑
    @File : img_embedding.py
    @desc : 使用pmc_clip的图像encoder对图像进行编码
"""


import os
import json
import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from open_clip import create_model_and_transforms, get_tokenizer


# Initialize the BiomedCLIP model
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Adjust GPU selection if needed


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


if (not model_name.startswith("hf_") and model_name not in _MODEL_CONFIGS and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg


# Initialize tokenizer, model, and preprocess function
tokenizer = get_tokenizer(model_name)
model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


def read_image(image_path):
    """Read an image from a file and preprocess it for the model."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)


def get_image_embedding(image_path):
    """Encode an image using BiomedCLIP and return the embedding."""
    image_tensor = read_image(image_path)
    with torch.no_grad():
        embeddings = model(image_tensor)[0]  # Extract embeddings
    return embeddings.cpu().numpy().squeeze()


def quantize_embedding(embedding, num_bits=8):
    """Quantize embedding to uint8."""
    min_val = np.min(embedding)
    max_val = np.max(embedding)
    scale = (max_val - min_val) / (2 ** num_bits - 1)
    quantized = np.round((embedding - min_val) / scale).astype(np.uint8)
    return quantized, min_val, scale


def preprocess_and_save_embeddings(data, base_image_dir, output_file):
    """
    Preprocess image embeddings, quantize them, and save to a .npz file for fast retrieval.
    """
    quantized_embeddings = []
    min_vals = []
    scales = []
    image_paths = []

    for entry in tqdm(data, desc="Processing images"):
        image_name = entry["image_name"]
        mode = image_name.split("_")[2].strip()
        image_path = os.path.join(base_image_dir, mode, f"{image_name}.jpg")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            # Get embedding and quantize
            embedding = get_image_embedding(image_path)
            quantized, min_val, scale = quantize_embedding(embedding)

            quantized_embeddings.append(quantized)
            min_vals.append(min_val)
            scales.append(scale)
            image_paths.append(image_path)
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")

    # Save all data to .npz
    np.savez(
        output_file,
        quantized_embeddings=np.array(quantized_embeddings, dtype=np.uint8),
        min_vals=np.array(min_vals, dtype=np.float32),
        scales=np.array(scales, dtype=np.float32),
        image_paths=np.array(image_paths)
    )
    print(f"Embeddings saved to {output_file}")


def load_preprocessed_embeddings(npy_file):
    """Load preprocessed embeddings from a .npz file."""
    data = np.load(npy_file, allow_pickle=True)
    quantized_embeddings = data["quantized_embeddings"]
    min_vals = data["min_vals"]
    scales = data["scales"]
    image_paths = data["image_paths"]
    return quantized_embeddings, min_vals, scales, image_paths


def dequantize_embedding(quantized, min_val, scale):
    """Dequantize uint8 embedding back to original scale."""
    return quantized.astype(np.float32) * scale + min_val


def find_most_similar(image_path, npy_file, top_k=1):
    """Find the most similar image embeddings."""
    query_embedding = get_image_embedding(image_path)
    quantized_embeddings, min_vals, scales, image_paths = load_preprocessed_embeddings(npy_file)

    # Dequantize all stored embeddings
    stored_embeddings = [
        dequantize_embedding(q, min_v, s)
        for q, min_v, s in zip(quantized_embeddings, min_vals, scales)
    ]

    # Compute cosine similarity with all embeddings
    similarities = np.array([
        np.dot(query_embedding, e) / (np.linalg.norm(query_embedding) * np.linalg.norm(e))
        for e in stored_embeddings
    ])

    # Find the top-k matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_matches = [
        (os.path.basename(image_paths[idx]).split(".")[0], similarities[idx])
        for idx in top_indices
    ]

    return top_matches


def main(data_json_path, base_image_dir, output_file):
    # Load metadata
    with open(data_json_path, "r") as f:
        data = json.load(f)

    # Preprocess and save embeddings
    preprocess_and_save_embeddings(data, base_image_dir, output_file)


if __name__ == "__main__":
    # Paths
    data_json_path = "../../data/ref/ROCOv2/all_merged_data.json"
    base_image_dir = "../../data/ref/ROCOv2"
    npy_file = "../../data/ref/ROCOv2/quantized_image_embeddings.npz"

    # main(data_json_path, base_image_dir, npy_file)

    # Example usage
    query_image_path = "CXR145_IM-0290-1001.png"

    # Find the most similar images
    top_k_matches = find_most_similar(query_image_path, npy_file, top_k=3)
    for match, similarity in top_k_matches:
        print(f"Match: {match}, Similarity: {similarity}")




