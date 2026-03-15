# _*_ coding: utf-8 _*_

"""
    @Time : 2024/9/19 20:31 
    @Author : smile 笑
    @File : chat_lora_model.py
    @desc :
"""


import torch
from torch import nn
from timm.models.vision_transformer import vit_base_patch16_224
from transformers import Qwen2ForCausalLM, AutoTokenizer
from .conversation import get_conv_template
from peft import LoraConfig, get_peft_model, TaskType
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from safetensors.torch import load_file
import argparse


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_image(image_file, input_size=224):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)

    return transform(image).unsqueeze(0)


class ChatInternQwenQAModel(nn.Module):
    def __init__(self, vision_model_path="../../../save/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/BiomedCLIP-vit-base-patch16_224.pth", torch_dtype=torch.float,
                 language_model_path="../../../save/Qwen2.5-0.5B-Instruct", lora_model=True,
                 load_model_path="../../../save/bio_qwen2_5_pre_lora_model_base/pre_best_model/bio_qwen2_5_pre_lora_model_base_no_freeze_vision/model_state_last.pth"):
        super(ChatInternQwenQAModel, self).__init__()

        vit_hidden_size = 768
        llm_hidden_size = 896
        self.num_image_token = 196
        self.template = "Hermes-2"
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.load_model_path = load_model_path

        self.vision_model = vit_base_patch16_224()
        self.vision_model.load_state_dict(torch.load(vision_model_path))
        self.language_model = Qwen2ForCausalLM.from_pretrained(language_model_path, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path, trust_remote_code=True, use_fast=False)

        if lora_model:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.language_model = get_peft_model(self.language_model, lora_config)  # 对llm模型加入lora，并冻结llm

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size, dtype=torch_dtype),
            nn.Linear(vit_hidden_size, llm_hidden_size, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=torch_dtype)
        )

    def parameters_replace(self):
        pretrained_state_dict = load_file(self.load_model_path)  # 使用 safetensors 来加载
        my_model_state_dict = self.state_dict()  # 获取原模型的参数名称

        original_keys = list(my_model_state_dict.keys())  # 获取原模型的参数名称
        pretrained_keys = list(pretrained_state_dict.keys())  # 获取新模型的参数名称

        # 逐一替换预训练模型的参数到原模型中
        for orig_key, target_key in zip(original_keys, pretrained_keys):
            if my_model_state_dict[orig_key].shape == pretrained_state_dict[target_key].shape:
                my_model_state_dict[orig_key] = pretrained_state_dict[target_key]

        self.load_state_dict(my_model_state_dict)
        print("参数更新完成!")

    def chat(self, pixel_values, question, generation_config, history=None, return_history=False,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        question = '<image>\n' + question

        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, visual_features=None,
                 generation_config=None, output_hidden_states=None, return_dict=None, **generate_kwargs):
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.mlp1(self.vision_model.forward_features(pixel_values)[:, 1:])
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model_path", default="../../../save/pre_trained_chat/model_state_last.pth")
    parser.add_argument("--test_image", default="../../../data/ref/rad/images/synpic21776.jpg")
    parser.add_argument("--question", default="Please give a caption of the image.")
    args = parser.parse_args()

    model = ChatInternQwenQAModel(load_model_path=args.load_model_path).cuda()
    model.parameters_replace()

    test_image = args.test_image
    pixel_values = load_image(test_image).to(torch.float).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=3, repetition_penalty=2.5)

    question = args.question

    response, history = model.chat(pixel_values, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')  # 还不错，加入lora进行微调后

