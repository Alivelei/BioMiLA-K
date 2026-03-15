# _*_ coding: utf-8 _*_

"""
    @Time : 2025/2/11 12:07 
    @Author : smile 笑
    @File : __init__.py
    @desc :
"""


def get_prompt_model_module(model_name):
    if model_name == "bio_qwen2_5_pre_caption_model_base":
        from .BioQwen2_5.pre_caption_model2 import bio_qwen2_5_pre_caption_model_base
        return bio_qwen2_5_pre_caption_model_base

    if model_name == "bio_qwen2_5_pre_lora_model_base":
        from .BioQwen2_5.pre_qa_lora2 import bio_qwen2_5_pre_lora_model_base
        return bio_qwen2_5_pre_lora_model_base


def get_model_module(model_name):
    if model_name == "bio_qwen2_5_vqa_model_base":
        from .BioQwen2_5.model import bio_qwen2_5_vqa_model_base
        return bio_qwen2_5_vqa_model_base
