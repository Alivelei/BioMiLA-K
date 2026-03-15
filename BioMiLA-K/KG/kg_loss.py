# _*_ coding: utf-8 _*_

"""
    @Time : 2025/2/9 16:35 
    @Author : smile 笑
    @File : kg_loss.py
    @desc :
"""

import json
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# 设置环境变量（如果需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class EnhancedSelectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout=0.1):
        """
        改进的 MLP 网络，优化参数初始化、激活函数和归一化策略。
        """
        super(EnhancedSelectionMLP, self).__init__()

        # 输入层
        self.fc_input = nn.Linear(in_dim, hidden_dim)

        # 扩展特征的 MLP 层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # BatchNorm 替代 LayerNorm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """
        使用更稳定的 Kaiming 初始化，提高训练稳定性。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播，输入特征 -> 预测得分。
        """
        # 输入层
        x = self.fc_input(x)
        x = self.ln1(x)
        x = F.gelu(x)  # 更平滑的非线性激活函数

        # 第一层 MLP
        x_post = self.fc1(x)
        x_post = self.ln1(x_post)
        x_post = F.gelu(x_post)

        # 第二层 MLP
        x_post = self.fc2(x_post)
        x_post = self.ln2(x_post)
        x_post = F.gelu(x_post)
        x_post = self.dropout(x_post)  # 在残差前 Dropout

        # 残差连接
        x = x + x_post

        # 输出层
        score = torch.sigmoid(self.fc_out(x))

        return score.squeeze(-1)

    def save_model(self, file_path):
        """
        保存模型权重。
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        加载模型权重。
        """
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")


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
    def __init__(self, mlp_hidden_dim=512, sim_threshold=0.1, margin=0.1, device=None, medclip_path="./BiomedCLIP/"):
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
        self.medclip_path = medclip_path

        # 下载并加载模型和配置文件
        # hf_hub_download(
        #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     filename="open_clip_pytorch_model.bin",
        #     local_dir="checkpoints"
        # )
        # hf_hub_download(
        #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        #     filename="open_clip_config.json",
        #     local_dir="checkpoints"
        # )
        with open(os.path.join(self.medclip_path, "./checkpoints/open_clip_config.json"), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

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
            pretrained=os.path.join(self.medclip_path, "./checkpoints/open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        self.model.to(self.device)
        self.model.eval()  # 固定 BiomedCLIP 模型

        # 创建候选文本选择的 MLP 模块，输入维度为图像embedding (512) 与文本embedding (512) 拼接后的 1024
        self.selection_mlp = EnhancedSelectionMLP(in_dim=1024, hidden_dim=mlp_hidden_dim).to(self.device)

    def get_embeddings_and_logits(self, image, candidate_texts):
        """
        给定图像（PIL 或预处理后的 tensor）和候选文本（list of str），
        利用 BiomedCLIP 模型计算图像和文本的 embedding 以及相似度得分 logits。实体只用前十个实体
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
        candidate_texts = candidate_texts[:10] if len(candidate_texts) > 0 else "None"
        texts = self.tokenizer(candidate_texts, context_length=256).to(self.device)
        with torch.no_grad():
            image_features, candidate_text_features, logit_scale = self.model(image, texts)
            # 计算 logits：使用 provided 代码片段
            logits = (logit_scale * image_features @ candidate_text_features.t()).detach().softmax(dim=-1)
        return image_features, candidate_text_features, logits

    def forward(self, image, candidate_texts):
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
          返回：
            loss_total：联合损失
            selected_entities：最终被选择的候选文本（列表）
        """
        # Step 1: 获取 BiomedCLIP 计算的 embedding 和 logits
        image_features, candidate_text_features, logits = self.get_embeddings_and_logits(image, candidate_texts)
        # 假设 logits 形状为 [1, N]，取第 0 行
        initial_scores = logits[0]  # [N]

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

        k = min(5, selection_prob.shape[0])  # 选取最多 k=5，但不能超过实际数量
        topk_values, topk_indices = selection_prob.topk(k=k, dim=0)
        if len(candidate_texts) > 0:
            # `topk_indices` 是 Tensor，需要转换为 Python 列表进行索引
            selected_entities = [candidate_texts[idx] for idx in topk_indices.tolist()]
        else:
            selected_entities = ["None"]

        return loss_cls, ranking_loss, selected_entities


if __name__ == '__main__':
    # 加载图像（假设为 PIL 图像）
    image_path = "./BiomedCLIP/CXR145_IM-0290-1001.png"
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

    # 创建 BiomedCLIPSelector 对象
    selector = BiomedCLIPSelector(mlp_hidden_dim=512, sim_threshold=0.7, margin=0.2)
    selector.to(selector.device)

    # 前向计算，返回总损失和最终被选择的实体文本列表
    loss_c, loss_r, selected_entities = selector(image_pil, candidate_texts)

    print("Loss_cls:", loss_c.item(), ", Loss_rank:", loss_r.item())
    print("Selected Entities (for prompt):", selected_entities)

    # 保存 MLP 模型
    selector.selection_mlp.save_model("enhanced_selection_mlp.pth")

    # 加载 MLP 模型
    loaded_mlp = EnhancedSelectionMLP(in_dim=1024)
    loaded_mlp.load_model("enhanced_selection_mlp.pth")


