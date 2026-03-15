# _*_ coding: utf-8 _*_

"""
    @Time : 2025/2/9 16:35 
    @Author : smile 笑
    @File : kg_loss.py
    @desc :
"""

import json
import os
from urllib.request import urlopen
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# 设置环境变量（如果需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


##############################
# 封装 BiomedCLIPSelector 类 #
##############################

class SelectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        """
        一个简单的 MLP，用于对图像与文本拼接特征进行二分类打分（0~1）
        """
        super(SelectionMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        score = torch.sigmoid(self.fc2(x))
        return score.squeeze(-1)  # 返回形状 [N]


class EnhancedSelectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        改进的 MLP 网络，包含多头注意力和非线性映射。
        """
        super(EnhancedSelectionMLP, self).__init__()

        # 输入层到隐藏层的全连接层
        self.fc_input = nn.Linear(in_dim, hidden_dim)

        # 多头注意力层（Multi-Head Attention）
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # 非线性映射层，增加模型表达能力
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层，生成对每个候选的预测得分
        self.fc_out = nn.Linear(hidden_dim, 1)

        # 残差连接（Residual Connection）
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播：图像嵌入与文本嵌入的拼接特征作为输入，输出文本实体选择的预测得分。
        """
        # Step 1: 输入层（全连接层）
        x = F.silu(self.fc_input(x))  # [N, hidden_dim]

        # Step 2: 多头注意力层（计算文本与图像的关系）
        # 注意力机制：self.attn(query, key, value)
        # query, key, value 都是 x，维度 [N, hidden_dim]
        attn_output, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_output = attn_output.squeeze(0)  # [N, hidden_dim]

        # Step 3: 非线性映射层
        x = F.silu(self.fc1(attn_output))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))

        # Step 4: 残差连接和层归一化
        x = self.layer_norm(x + attn_output)

        # Step 5: 输出层（生成最终预测得分）
        score = torch.sigmoid(self.fc_out(x))  # [N, 1]

        return score.squeeze(-1)  # 返回形状为 [N] 的得分向量


def gumbel_softmax(logits, tau=0.5, hard=True):
    """
    Gumbel-Softmax 采样，将连续 logits 近似为离散选择向量
    """
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + noise) / tau, dim=0)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard[y.argmax()] = 1.0
        y = (y_hard - y).detach() + y
    return y


class BiomedCLIPSelector(nn.Module):
    def __init__(self, mlp_hidden_dim=128, sim_threshold=0.1, margin=0.1, device=None):
        """
        封装了 BiomedCLIP 模型、tokenizer、预处理函数及候选文本选择模块。
        参数：
          mlp_hidden_dim：MLP 模块隐藏层维度
          sim_threshold：利用 BiomedCLIP 计算相似度得分生成伪标签的阈值
          margin：排序损失中正负候选得分的 margin
          device：计算设备（默认为 cuda 如果可用，否则 cpu）
        """
        super(BiomedCLIPSelector, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sim_threshold = sim_threshold
        self.margin = margin

        # 下载并加载模型和配置文件
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
        with open("checkpoints/open_clip_config.json", "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        # 如果模型名称不在 _MODEL_CONFIGS 中，则添加
        model_name = "biomedclip_local"
        if (not model_name.startswith(HF_HUB_PREFIX)
                and model_name not in _MODEL_CONFIGS
                and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        # 加载 tokenizer
        self.tokenizer = get_tokenizer(model_name)

        # 创建模型和预处理函数
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained="checkpoints/open_clip_pytorch_model.bin",
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        self.model.to(self.device)
        self.model.eval()  # 固定 BiomedCLIP 模型

        # 创建候选文本选择的 MLP 模块，输入维度为图像embedding (512) 与文本embedding (512) 拼接后的 1024
        self.selection_mlp = EnhancedSelectionMLP(in_dim=1024, hidden_dim=mlp_hidden_dim).to(self.device)

    def get_embeddings_and_logits(self, image, candidate_texts):
        """
        给定图像（PIL 或预处理后的 tensor）和候选文本（list of str），
        利用 BiomedCLIP 模型计算图像和文本的 embedding 以及相似度得分 logits。
        使用你提供的代码片段计算 logits：
          with torch.no_grad():
              image_features, text_features, logit_scale = model(image, texts)
              logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        返回：
          image_features: [1, 512] 的图像 embedding
          candidate_text_features: [N, 512] 的候选文本 embedding
          logits: [1, N] 的相似度得分（softmax 后），用作伪标签生成
        """
        # 如果 image 不是 tensor，则利用 preprocess 处理（假设 image 为 PIL 图像）
        if not torch.is_tensor(image):
            image = self.preprocess(image)
            image = image.unsqueeze(0)  # [1, C, H, W]
        image = image.to(self.device)

        # 对候选文本进行编码
        texts = self.tokenizer(candidate_texts, context_length=256).to(self.device)

        with torch.no_grad():
            image_features, candidate_text_features, logit_scale = self.model(image, texts)
            # 计算 logits：使用 provided 代码片段
            # image_features: [1,512]； candidate_text_features: [N,512]； logit_scale: 标量或 [1]
            logits = (logit_scale * image_features @ candidate_text_features.t()).detach().softmax(dim=-1)
        return image_features, candidate_text_features, logits

    def forward(self, image, candidate_texts, L_CE=None):
        """
        主前向函数：
          输入：
            image：输入图像（PIL 图像或 tensor）
            candidate_texts：候选文本列表（list of str），包含实体和关系信息
            L_CE：（可选）下游大模型的交叉熵损失
          过程：
            1. 利用 BiomedCLIP 计算图像 embedding、候选文本 embedding 及相似度得分 logits
            2. 以 logits 为依据生成伪标签（得分 > sim_threshold 标记为正）
            3. 构造图像与候选文本拼接特征，输入 MLP 得到候选预测得分
            4. 计算二分类损失（BCE）和 margin 排序损失
            5. 使用 Gumbel-Softmax 近似离散选择候选（例如选择得分最高的候选）
            6. 将候选损失与 L_CE 联合（如果传入）形成总损失
          返回：
            loss_total：联合损失
            selected_entities：最终被选择的候选文本（列表）
        """
        # Step 1: 获取 BiomedCLIP 计算的 embedding 和 logits
        image_features, candidate_text_features, logits = self.get_embeddings_and_logits(image, candidate_texts)
        # 假设 logits 形状为 [1, N]，取第 0 行
        initial_scores = logits[0]  # [N]
        print(initial_scores)
        # Step 2: 根据 initial_scores 生成伪标签：得分大于 sim_threshold 标记为正 (1)，否则为负 (0)
        pseudo_labels = (initial_scores > self.sim_threshold).float()  # [N]

        # Step 3: 构造拼接特征：将 image_features 重复 N 次并与 candidate_text_features 拼接
        N = candidate_text_features.size(0)
        image_features_rep = image_features.expand(N, -1)  # [N, 512]
        combined_features = torch.cat([image_features_rep, candidate_text_features], dim=-1)  # [N, 1024]

        # Step 4: 通过 MLP 得到候选预测得分（范围 [0,1]）
        mlp_scores = self.selection_mlp(combined_features)  # [N]

        # Step 5: 计算二分类损失（BCE Loss），指导 MLP 学习区分正负候选
        loss_cls = F.binary_cross_entropy(mlp_scores, pseudo_labels)

        # Step 6: 构造 margin 排序损失：令正例得分与负例得分间至少差 margin
        sorted_scores, sorted_indices = torch.sort(mlp_scores, descending=True)
        pos_count = int((pseudo_labels > 0.5).sum().item())
        if pos_count < 1:
            pos_count = 1
        pos_scores = sorted_scores[:pos_count]
        neg_scores = sorted_scores[pos_count:]
        ranking_loss = 0.0
        if neg_scores.numel() > 0:
            for sp in pos_scores:
                diff = sp - neg_scores  # [num_neg]
                ranking_loss += F.relu(self.margin - diff).mean()
            ranking_loss = ranking_loss / pos_count
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)

        # Step 7: 使用 Gumbel-Softmax 将 mlp_scores 近似为离散选择向量
        selection_prob = gumbel_softmax(mlp_scores, tau=0.5, hard=True)  # [N]
        # 这里简单地选择得分最高的候选（即 selection_prob 中值为1的索引）
        selected_index = selection_prob.argmax().item()
        selected_entities = [candidate_texts[selected_index]]

        # Step 8: 总候选损失
        lambda_cls = 1.0
        lambda_rank = 1.0
        loss_candidates = lambda_cls * loss_cls + lambda_rank * ranking_loss

        # Step 9: 联合下游大模型损失（如果传入 L_CE），否则只使用候选损失
        if L_CE is not None:
            loss_total = L_CE + loss_candidates
        else:
            loss_total = loss_candidates

        return loss_total, selected_entities


#####################################
# 示例：调用 BiomedCLIPSelector 类进行前向计算
#####################################

if __name__ == '__main__':
    # 加载图像（假设为 PIL 图像）
    image_path = "CXR145_IM-0290-1001.png"
    image_pil = Image.open(image_path)

    # 定义候选文本实体和关系列表（可从知识图谱中检索到的候选）
    candidate_texts = [
        'adenocarcinoma histopathology',
        'brain MRI',
        'covid line chart',
        'chest X-Ray',
        'no evidence of pneumonia',
        'benign nodule detected'
    ]

    # 假设下游大模型损失 L_CE 目前没有传入，置为 None
    L_CE = None

    # 创建 BiomedCLIPSelector 对象
    selector = BiomedCLIPSelector(mlp_hidden_dim=128, sim_threshold=0.7, margin=0.2)
    selector.to(selector.device)

    # 前向计算，返回总损失和最终被选择的实体文本列表
    loss_total, selected_entities = selector.forward(image_pil, candidate_texts, L_CE=L_CE)

    print("Total Loss:", loss_total.item())
    print("Selected Entities (for prompt):", selected_entities)

