# _*_ coding: utf-8 _*_

"""
    @Time : 2025/2/10 17:10 
    @Author : smile 笑
    @File : kg_main.py
    @desc :
"""


from .kg_match import KGMultiModalMatch, query_image_relations
from .kg_loss import BiomedCLIPSelector
from torch import nn
from PIL import Image


class KGController(nn.Module):
    def __init__(self, embedding_npy_file="../data/ref/ROCOv2/quantized_image_embeddings.npz",
                 medclip_path="./BiomedCLIP/", top_k=3, mlp_hidden_dim=512, sim_threshold=0.2, margin=0.2):
        super(KGController, self).__init__()
        self.embedding_npy_file = embedding_npy_file
        self.top_k = top_k
        self.kg_match = KGMultiModalMatch(medclip_path=medclip_path, npy_file=embedding_npy_file)
        self.selector = BiomedCLIPSelector(mlp_hidden_dim=mlp_hidden_dim, sim_threshold=sim_threshold, margin=margin,
                                           medclip_path=medclip_path)

    def save_mlp_model(self, save_path):
        # 保存 MLP 模型
        self.selector.selection_mlp.save_model(save_path)

    def load_mlp_model(self, save_path):
        # 加载 MLP 模型
        self.selector.selection_mlp.load_model(save_path)

    def forward(self, image_path):
        candidate_texts = []
        top_k_matches = self.kg_match.find_most_similar(image_path, top_k=self.top_k)

        for match, similarity in top_k_matches:
            results = query_image_relations(self.kg_match.graph, match)
            for data in results:
                candidate_texts.append(data["entity"] + "->" + data["relation_type"] + "->" + data["related_entity"])

        image_pil = Image.open(image_path)
        # 前向计算，返回总损失和最终被选择的实体文本列表
        loss_c, loss_r, selected_entities = self.selector(image_pil, candidate_texts)

        return loss_c, loss_r, selected_entities


if __name__ == '__main__':
    import torch.optim as optim
    from PIL import Image

    QUERY_IMAGE_PATH = "../data/ref/ROCOv2/test/ROCOv2_2023_test_003600.jpg"
    model = KGController()
    # print(model(QUERY_IMAGE_PATH))

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(5):
        total_loss = 0
        for _ in range(5):
            # 生成随机数据
            image_pil = QUERY_IMAGE_PATH

            # 前向计算
            loss_c, loss_r, selected_entities = model(image_pil)
            loss = loss_c + loss_r

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印训练进度
        print(f"Epoch {epoch + 1}/{5}, Loss: {total_loss / 5:.4f}")

    print("训练完成！")



