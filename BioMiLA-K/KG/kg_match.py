# _*_ coding: utf-8 _*_

"""
    @Time : 2025/1/21 18:05 
    @Author : smile 笑
    @File : kg_match.py
    @desc :
"""


import os
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from open_clip.factory import _MODEL_CONFIGS
from open_clip import create_model_and_transforms
from .kg_create import query_image_relations, connect_to_neo4j


"""Initialize the BiomedCLIP model."""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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


class KGMultiModalMatch(object):
    def __init__(self, medclip_path="./BiomedCLIP/", npy_file="./"):
        self.medclip_path = medclip_path

        # Connect to Neo4j knowledge graph
        print("Connecting to Neo4j knowledge graph...")
        self.graph = connect_to_neo4j(uri="http://localhost:7474", auth=("neo4j", "zxc123zxc123.."), name="neo4j")

        # Initialize BiomedCLIP model
        print("Initializing BiomedCLIP model...")
        self.initialize_biomedclip()

        quantized_embeddings, min_vals, scales, self.image_paths = load_preprocessed_embeddings(npy_file)

        # Dequantize all stored embeddings
        self.stored_embeddings = [
            dequantize_embedding(q, min_v, s)
            for q, min_v, s in zip(quantized_embeddings, min_vals, scales)
        ]

    def initialize_biomedclip(self):
        # Download the model and config files
        # hf_hub_download(
        #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     filename="open_clip_pytorch_model.bin",
        #     local_dir=os.path.join(self.medclip_path, "/checkpoints/")
        # )
        # hf_hub_download(
        #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     filename="open_clip_config.json",
        #     local_dir=os.path.join(self.medclip_path, "/checkpoints/")
        # )

        # Load the model and config files
        model_name = "biomedclip_local"
        with open(os.path.join(self.medclip_path, "./checkpoints/open_clip_config.json"), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        if model_name not in _MODEL_CONFIGS and config is not None:
            _MODEL_CONFIGS[model_name] = model_cfg

        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=os.path.join(self.medclip_path, "./checkpoints/open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    def read_image(self, image_path):
        """Read an image from a file and preprocess it for the model."""
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def get_image_embedding(self, image_path):
        """Encode an image using BiomedCLIP and return the embedding."""
        image_tensor = self.read_image(image_path)
        with torch.no_grad():
            embeddings = self.model(image_tensor)[0]  # Extract embeddings
        return embeddings.cpu().numpy().squeeze()

    def find_most_similar(self, image_path, top_k=1):
        """Find the most similar image embeddings."""
        query_embedding = self.get_image_embedding(image_path)

        # Compute cosine similarity with all embeddings
        similarities = np.array([
            np.dot(query_embedding, e) / (np.linalg.norm(query_embedding) * np.linalg.norm(e))
            for e in self.stored_embeddings
        ])

        # Find the top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_matches = [
            (os.path.basename(self.image_paths[idx]).split(".")[0], similarities[idx])
            for idx in top_indices
        ]

        return top_matches


if __name__ == "__main__":
    # Example usage
    QUERY_IMAGE_PATH = "./BiomedCLIP/CXR145_IM-0290-1001.png"
    NPY_FILE = "../data/ref/ROCOv2/quantized_image_embeddings.npz"
    TOP_K = 3

    """Main function to find similar images and query knowledge graph."""
    kg_match = KGMultiModalMatch(NPY_FILE)

    print(f"Finding top {TOP_K} matches for {QUERY_IMAGE_PATH}...")
    top_k_matches = kg_match.find_most_similar(QUERY_IMAGE_PATH, top_k=TOP_K)
    print(top_k_matches)
    for match, similarity in top_k_matches:
        print(f"Match: {match}, Similarity: {similarity}")

        # Query relations for the most similar image
        print("Querying knowledge graph for image relations...")
        results = query_image_relations(kg_match.graph, match)
        print(f"Image: {match}, Relations: {results}")
        for data in results:
            print(data["entity"], data["relation_type"], data["related_entity"])

