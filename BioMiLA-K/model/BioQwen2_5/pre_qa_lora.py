# _*_ coding: utf-8 _*_

"""
    @Time : 2025/3/11 10:24 
    @Author : smile 笑
    @File : pre_qa_lora2.py
    @desc :
"""


import torch
from torch import nn
from transformers import AutoModel, Qwen2ForCausalLM, AutoTokenizer
from timm.models.vision_transformer import vit_base_patch16_224
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from transformers import PreTrainedModel, PretrainedConfig
from .conversation import get_conv_template


class BioMiLaConfig(PretrainedConfig):
    model_type = "BioMiLA-K"
    vision_model_path = "./save/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/BiomedCLIP-vit-base-patch16_224.pth"
    torch_dtype = torch.float
    language_model_path = "./save/Qwen2.5-0.5B-Instruct"
    freeze_vision_model = True
    freeze_language_model = True
    load_model_path = "./save/bio_qwen2_5_pre_caption_model_base/checkpoint-5300/model.safetensors"


class PreBioQwenLoraQAModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config=config)

        vit_hidden_size = 768
        llm_hidden_size = 896
        self.num_image_token = 196
        self.template = "Hermes-2"
        self.torch_dtype = config.torch_dtype
        self.load_model_path = config.load_model_path
        self.freeze_vision_model = config.freeze_vision_model

        # self.vision_model = InterVitModel(vision_model_path, torch_dtype=torch.float16)
        self.vision_model = vit_base_patch16_224()
        self.vision_model.load_state_dict(torch.load(config.vision_model_path))
        self.language_model = Qwen2ForCausalLM.from_pretrained(config.language_model_path, torch_dtype=config.torch_dtype, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_path, trust_remote_code=True, use_fast=False)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.language_model = get_peft_model(self.language_model, lora_config)  # 对llm模型加入lora，并冻结llm，微调lora

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size, dtype=config.torch_dtype),
            nn.Linear(vit_hidden_size, llm_hidden_size, dtype=config.torch_dtype),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=config.torch_dtype)
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

        # 冻结视觉模型，在第二阶段只微调mlp和lora看看效果
        if self.freeze_vision_model:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False

    def batch_merge(self, pixel_values, questions, labels, IMG_START_TOKEN='<img>',
                    IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        num_patches_list = [1 for _ in range(len(questions))]  # 仅支持单张图

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            label = labels[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], label)
            query = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)

        input_ids = model_inputs["input_ids"]
        input_labels = torch.full_like(input_ids, -100, dtype=torch.int64)
        for i in range(input_labels.shape[0]):
            label_idx = torch.where(input_ids[i] == 151644)[0][2]
            input_labels[i, label_idx + 3:] = input_ids[i, label_idx + 3:]

        return {
            "input_ids": model_inputs["input_ids"].cuda(),
            "attention_mask": model_inputs["attention_mask"].cuda(),
            "labels": input_labels.cuda(),
        }

    def get_language_embedding(self, input_ids, vit_embeds):
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        return input_embeds.reshape(B, N, C)

    def pool_features(self, hidden_states, input_ids, start_token_id, end_token_id):
        pooled_features = []
        for i in range(hidden_states.size(0)):
            start_idx = torch.where(input_ids[i] == start_token_id)[0]
            end_idx = torch.where(input_ids[i] == end_token_id)[0]
            # 对该段进行池化
            pooled_feature = hidden_states[i, start_idx:end_idx, :].mean(dim=0)  # 求平均池化
            pooled_features.append(pooled_feature)

        # 堆叠所有的池化特征
        return torch.stack(pooled_features, dim=0)

    def pool_entity_features(self, hidden_states, input_ids, entity_start_token_id, entity_end_token_id):
        all_batch_entity_features = []
        for i in range(hidden_states.size(0)):
            entity_features = []
            entity_start_idx = torch.where(input_ids[i] == entity_start_token_id)[0]
            entity_end_idx = torch.where(input_ids[i] == entity_end_token_id)[0]
            for idx in range(entity_start_idx.size(0)):
                # 对每个实体进行池化
                pooled_entity_feature = hidden_states[i, entity_start_idx[idx]:entity_end_idx[idx], :].mean(dim=0)
                entity_features.append(pooled_entity_feature)
            all_batch_entity_features.append(entity_features)
        return all_batch_entity_features

    def compute_entity_attention_loss(self, img_global_features, question_global_features, entity_global_features):
        batch_size = len(entity_global_features)
        att_total_loss = 0.0
        num_entities_total = 0  # 统计所有样本的实体总数

        for i in range(batch_size):  # 遍历每个数据样本
            img_feat = img_global_features[i]  # 当前样本的图像全局特征, shape = [D]
            text_feat = question_global_features[i]  # 当前样本的文本全局特征, shape = [D]

            entities = entity_global_features[i]  # 当前样本的实体特征列表

            if len(entities) == 0:  # 如果当前样本没有实体，跳过
                continue

            entity_attn_scores = []
            entity_attn_labels = []

            for entity_feat in entities:  # 遍历当前样本的所有实体
                # 计算实体与图像的点积注意力，并 softmax 归一化
                img_attn_score = F.softmax(torch.dot(entity_feat, img_feat))  # 标量
                text_attn_score = F.softmax(torch.dot(entity_feat, text_feat))  # 标量

                # 计算最终注意力得分
                attn_score = (img_attn_score + text_attn_score) / 2
                entity_attn_scores.append(attn_score)

                # 计算实体与图像的余弦相似度
                img_cos_sim = F.cosine_similarity(entity_feat.unsqueeze(0), img_feat.unsqueeze(0)).item()
                text_cos_sim = F.cosine_similarity(entity_feat.unsqueeze(0), text_feat.unsqueeze(0)).item()

                # 计算最终注意力标签
                attn_label = (img_cos_sim + text_cos_sim) / 2
                entity_attn_labels.append(attn_label)

            # 转换为张量
            entity_attn_scores = torch.tensor(entity_attn_scores, dtype=self.torch_dtype, device=img_feat.device)
            entity_attn_labels = torch.tensor(entity_attn_labels, dtype=self.torch_dtype, device=img_feat.device)

            # 计算 MSE 损失
            loss = F.mse_loss(entity_attn_scores, entity_attn_labels)
            att_total_loss += loss
            num_entities_total += len(entities)

        # 归一化 batch 内的总 loss
        if num_entities_total > 0:
            att_total_loss /= num_entities_total

        return att_total_loss

    def forward(self, images, texts, labels):
        images = torch.stack(images).squeeze() if not isinstance(images, torch.Tensor) else images
        images = images.unsqueeze(0) if len(images.shape) == 3 else images

        inputs = self.batch_merge(images, texts, labels)

        # 提取图像特征
        med_vit_embeds = self.vision_model.forward_features(images)[:, 1:]
        vit_embeds = self.mlp1(med_vit_embeds)

        # 替换进 language_model 的输入
        inputs_embeds = self.get_language_embedding(inputs["input_ids"], vit_embeds)

        # 计算 language_model 的输出
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"],
                                      labels=inputs["labels"], output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 取最后一层特征

        # 计算图像全局特征（从 <img> 到 </img> 之间进行pooling）
        img_start_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        img_end_token_id = self.tokenizer.convert_tokens_to_ids("</img>")
        img_global_feature = self.pool_features(hidden_states, inputs["input_ids"], img_start_token_id,
                                                img_end_token_id).cuda()

        # 计算问题全局特征（从 <QUS> 到 </QUS> 之间进行pooling）
        qus_start_token_id = self.tokenizer.convert_tokens_to_ids("<QUS>")
        qus_end_token_id = self.tokenizer.convert_tokens_to_ids("</QUS>")
        question_global_feature = self.pool_features(hidden_states, inputs["input_ids"], qus_start_token_id,
                                                     qus_end_token_id).cuda()

        # 假设实体的标记是 <ENTITY> 和 </ENTITY>
        entity_start_token_id = self.tokenizer.convert_tokens_to_ids("<ENT>")
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids("</ENT>")

        # 为每个实体执行池化操作
        entity_global_features = self.pool_entity_features(hidden_states, inputs["input_ids"], entity_start_token_id, entity_end_token_id)

        att_loss = self.compute_entity_attention_loss(img_global_feature, question_global_feature, entity_global_features)

        return outputs.loss, att_loss, outputs.logits


def bio_qwen2_5_pre_lora_model_base(**kwargs):
    return PreBioQwenLoraQAModel(BioMiLaConfig())


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    selected_entity = "<ENT>Head CT -> Head CT -> Head CT</ENT>, <ENT>Head CT -> Head CT -> Head CT</ENT>"
    text = [
        f"Relevant matching entity relationships found through the knowledge base: {selected_entity} \n" + f"<QUS>Please describe this image in detail based on the image, about the images.</QUS>",
        f"Relevant matching entity relationships found through the knowledge base: {selected_entity} \n" + f"<QUS>Please describe this image in detail based on the image.</QUS>"]

    label = [
        "The image shows a close-up of a brain with a large black spot in the center. ",
        "A fish the image shows a close-up of a brain with a large black and white section, which is a hyperintensity. This is caused by ischemia, which is a lack of blood flow to the brain."
    ]
    image = torch.rand([2, 3, 224, 224]).cuda()

    # model = QwenQAPreModel().cuda()
    # model(text, label)
    model = PreBioQwenLoraQAModel(BioMiLaConfig()).cuda()
    ls1, ls2, ot = model(image, text, label)
    print(sum(x.numel() for x in model.parameters()))  # 584257384  多了2百万参数量
    print(ls1, ls2, ot)


