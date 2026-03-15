# _*_ coding: utf-8 _*_

"""
    @Time : 2025/3/9 10:57 
    @Author : smile 笑
    @File : pre_train_first4.py
    @desc :
"""


import argparse
import os
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, set_seed
from data import PreCaptionROCOV2Dataset  # 确保数据集返回 dict 格式的数据
from model import get_prompt_model_module
from KG.kg_main import KGController


def collate_fn_qa(batch):
    image, qus, ans, image_paths = list(zip(*batch))
    image = torch.stack(image, dim=0).squeeze()
    return {
        "images": image,
        "texts": qus,
        "labels": ans,
        "image_paths": image_paths,
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
            inputs["texts"][i] = f"\n Relevant matching entity relationships found through the knowledge base: {selected_entity} \n" + "Question: <QUS>"+inputs["texts"][i]+"</QUS>"

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


def main(args):
    # 设置随机种子
    set_seed(args.random_seed)

    # 构造模型名称并实例化模型
    model_name = f"{args.model_select}_{args.model_size}"

    # 获取模型构造函数（返回的是模型类）
    model = get_prompt_model_module(model_name)()

    # 定义保存目录
    output_dir = os.path.join(args.default_root_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # 构造 TrainingArguments，添加混合精度、预热、梯度裁剪、多线程加载等优化措施
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weights_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=2,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="no",  # 如有验证集可设置为 "steps" 或 "epoch"
        seed=args.random_seed,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
    )

    # 构建优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weights_decay,
        eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)

    # 实例化数据集（要求返回 image, question, answer, image_path 四个元素）
    train_dataset = PreCaptionROCOV2Dataset(args)

    kgc = KGController(embedding_npy_file="./data/ref/ROCOv2/quantized_image_embeddings.npz",
                       medclip_path="./KG/BiomedCLIP/", top_k=2, mlp_hidden_dim=1024, sim_threshold=0.05, margin=0.1)

    if os.path.exists(os.path.join(output_dir, args.kg_mlp_model_path)):
        kgc.load_mlp_model(os.path.join(output_dir, args.kg_mlp_model_path))

    # 使用自定义 KGTrainer
    trainer = KGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_qa,
        optimizers=(optimizer, scheduler),
        kgc=kgc,
    )
    trainer.train()

    # 训练结束后保存知识库内部的 MLP 模型
    kgc.save_mlp_model(os.path.join(output_dir, args.kg_mlp_model_path))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    # 模型相关参数
    parser.add_argument("--model_select", default="bio_qwen2_5_pre_caption_model",
                        choices=["bio_qwen2_5_pre_caption_model"])
    parser.add_argument("--model_size", default="base", choices=["tiny", "base", "large"])

    # 知识库相关参数
    parser.add_argument("--embedding_npy_file", default="./data/ref/ROCOv2/quantized_image_embeddings.npz")
    parser.add_argument("--medclip_path", default="./KG/BiomedCLIP")
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--mlp_hidden_dim", default=512, type=int)
    parser.add_argument("--sim_threshold", default=0.05, type=float)
    parser.add_argument("--margin", default=0.1, type=float)
    parser.add_argument("--kg_mlp_model_path", default="kg_selector.bin")

    # 基础训练参数
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--weights_decay", default=0.05, type=float)
    parser.add_argument("--random_seed", default=1024, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--grad_accumulation_steps", default=4, type=int)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--warmup_steps", default=250, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # 图像处理相关参数
    parser.add_argument("--img_rotation", default=15, type=int)
    parser.add_argument("--resized_crop_left", default=0.2, type=float)
    parser.add_argument("--resized_crop_right", default=0.2, type=float)
    parser.add_argument("--image_size", default=224, type=int)

    # 文件保存与恢复相关参数
    parser.add_argument("--default_root_dir", default="./save/")

    # 数据集相关参数
    parser.add_argument("--rocov2_merged_data_path", default="./data/ref/ROCOv2/all_merged_data.json")
    parser.add_argument("--rocov2_filtered_instruct_data", default="./data/ref/ROCOv2/rocov2_filtered_instruct.json")
    parser.add_argument("--all_pretrained_data_path", default="./data/ref/ROCOv2/")

    args = parser.parse_args()
    main(args)

