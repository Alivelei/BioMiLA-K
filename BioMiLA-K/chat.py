# _*_ coding: utf-8 _*_

"""
    @Time : 2025/3/13 11:35 
    @Author : smile 笑
    @File : chat.py
    @desc :
"""


import argparse
import os
import torch
from torch import nn
from transformers import Trainer, TrainingArguments 
from KG.kg_main import KGController
from model.BioQwen2_5.chat_lora_model import ChatInternQwenQAModel, load_image


# --- 自定义 collate_fn，返回图像、问题、答案和图像路径 ---
def collate_fn_qa(batch):
    images, qus, ans, image_paths = list(zip(*batch))
    images = torch.stack(images, dim=0).squeeze()
    return {
        "images": images,
        "texts": qus,
        "labels": ans,
        "image_paths": image_paths
    }


# --- 自定义 KGTrainer，重写 compute_loss 来组合模型损失和知识库损失 ---
class KGTrainer(Trainer):
    def __init__(self, *args, kgc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kgc = kgc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kg_loss_c = 0.0
        kg_loss_r = 0.0

        for i, img_path in enumerate(inputs["image_paths"]):
            # 通过知识库获取实体
            loss_c, loss_r, selected_entity = self.kgc(img_path)
            inputs["texts"][i] = f"\n Relevant matching entity relationships found through the knowledge base: {selected_entity} \n please answer the question accurately based on image and entity relationships. \n" + "Question: <QUS>"+inputs["texts"][i]+"</QUS>"

            # 在文本中标注实体
            for entity in selected_entity:
                inputs["texts"][i] = inputs["texts"][i].replace(entity, f"<ENT>{entity}</ENT>")

            kg_loss_c += loss_c
            kg_loss_r += loss_r

        # 计算模型原始损失
        n = len(inputs["image_paths"])
        model_loss, att_loss, _ = model(inputs["images"], inputs["texts"], inputs["labels"])  # 这里只传文本，不使用 tokenizer

        kg_loss = (kg_loss_c + kg_loss_r) / n
        total_loss = model_loss + kg_loss + att_loss

        if self.state.global_step % 50 == 0:
            self.log({
                "total_loss": total_loss.item(),
                "kg_loss_c": kg_loss_c.item(),
                "kg_loss_r": kg_loss_r.item(),
                "att_loss": att_loss.item(),
                "loss_model": model_loss.item(),
            })

        return total_loss


class ChatBioMiLAK(nn.Module):
    def __init__(self, args):
        super(ChatBioMiLAK, self).__init__()

        self.kgc = KGController(embedding_npy_file="./data/ref/ROCOv2/quantized_image_embeddings.npz",
                                medclip_path="./KG/BiomedCLIP/", top_k=2, mlp_hidden_dim=1024, sim_threshold=0.05,
                                margin=0.1)
        self.model = ChatInternQwenQAModel(
            load_model_path=args.load_model_path,
            vision_model_path="./save/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/BiomedCLIP-vit-base-patch16_224.pth",
            torch_dtype=torch.float,
            language_model_path="./save/Qwen2.5-0.5B-Instruct", lora_model=True,
        )
        self.model.parameters_replace()

    def forward(self, image_path, question):
        pixel_values = load_image(image_path).to(torch.float).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=3, repetition_penalty=2.5)

        loss_c, loss_r, selected_entity = self.kgc(image_path)
        # question = f"\n Relevant matching entity relationships found through the knowledge base: {selected_entity} \n please answer the question accurately based on image and entity relationships. \n" + "Question: <QUS>" + question + "</QUS>"

        response, history = self.model.chat(pixel_values, question, generation_config, history=None, return_history=True)
        print(f'User: {question}\nAssistant: {response}')  # 还不错，加入lora进行微调后

        return


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model_path", default="./save/bio_qwen2_5_pre_lora_model_base/checkpoint-3600/model.safetensors")
    parser.add_argument("--test_image", default="./data/ref/rad/images/synpic21776.jpg")
    parser.add_argument("--question", default="Please give a caption of the image.")
    args = parser.parse_args()

    ChatBioMiLAK(args).to("cuda")(args.test_image, args.question)


