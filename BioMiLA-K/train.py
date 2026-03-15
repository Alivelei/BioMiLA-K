# _*_ coding: utf-8 _*_

"""
    @Time : 2024/9/20 14:31 
    @Author : smile 笑
    @File : train.py
    @desc :
"""


import argparse
from data import DataInterfaceModule, SlakeDatasetModule, RadDatasetModule, PathVQADatasetModule, OVQADatasetModule
from data import LoadWsSlakeDatasetModule, LoadWsRadDatasetModule, LoadWsOVQADatasetModule, LoadWsPathVQADatasetModule
from framework import ModelInterfaceModule, LoadWsModelInterfaceModule, get_model_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def mkdir_println(dir_path, println):
    if os.path.exists(dir_path):
        print(println + "文件夹已创建.")
    else:
        os.mkdir(dir_path)
        print(println + "文件夹创建成功.")


def create_model_module(args):
    model_name = args.model_select + "_" + args.model_size
    model_func = get_model_module(model_name)

    if args.load_ws_answer:
        model = LoadWsModelInterfaceModule(model=model_func, args=args)
    else:
        model = ModelInterfaceModule(model=model_func, args=args)

    args.default_root_dir = os.path.join(args.default_root_dir, model_name + "/")
    mkdir_println(args.default_root_dir, model_name + "根")  # 创建模型根文件夹

    args.train_epoch_effect_path = os.path.join(args.default_root_dir, args.train_epoch_effect_path)
    args.test_epoch_effect_path = os.path.join(args.default_root_dir, args.test_epoch_effect_path)
    mkdir_println(args.train_epoch_effect_path, model_name + "_param")  # 创建根文件夹下的param

    args.best_model_path = os.path.join(args.default_root_dir, args.best_model_path)
    mkdir_println(args.best_model_path, model_name + "_train_best_model")  # 创建根文件夹下的训练集最佳模型文件夹

    args.test_best_model_path = os.path.join(args.default_root_dir, args.test_best_model_path)
    mkdir_println(args.test_best_model_path, model_name + "_test_best_model")  # 创建根文件夹下测试集最佳模型文件夹

    return model, args


def dataset_select(args):
    if args.load_ws_answer:
        if args.select_data == "slake":
            db = DataInterfaceModule(LoadWsSlakeDatasetModule, args)
        if args.select_data == "rad":
            db = DataInterfaceModule(LoadWsRadDatasetModule, args)
        if args.select_data == "path_vqa":
            db = DataInterfaceModule(LoadWsPathVQADatasetModule, args)
        if args.select_data == "ovqa":
            db = DataInterfaceModule(LoadWsOVQADatasetModule, args)
    else:
        if args.select_data == "slake":
            db = DataInterfaceModule(SlakeDatasetModule, args)
        if args.select_data == "rad":
            db = DataInterfaceModule(RadDatasetModule, args)
        if args.select_data == "path_vqa":
            db = DataInterfaceModule(PathVQADatasetModule, args)
        if args.select_data == "ovqa":
            db = DataInterfaceModule(OVQADatasetModule, args)

    # 用来获取是哪个版本的模型
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.model_select + "_" + args.model_size + "_" + args.select_data + "_" + str(args.version),
        name="train_logs"
    )

    return db, logger


def main(args):
    seed_everything(args.random_seed, True)  # 设置随机数种子

    model, args = create_model_module(args)
    db, logger = dataset_select(args)

    train_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=0,
        dirpath=os.path.join(args.best_model_path, str(logger.version)),
        filename="{epoch}-{train_loss:.4f}",
        save_last=True
    )

    test_checkpoint_callback = ModelCheckpoint(
        monitor="test_total_acc",
        mode="max",
        save_top_k=1,
        dirpath=os.path.join(args.test_best_model_path, str(logger.version)),
        filename="{epoch}-{test_total_acc:.4f}",
        save_weights_only=True,
        # save_last=True
    )

    # 构建json保存路径
    epoch_effect_path = os.path.join(args.train_epoch_effect_path, str(logger.version))
    mkdir_println(epoch_effect_path, "model_param_version")  # 创建param下的version文件夹
    args.train_epoch_effect_path = os.path.join(epoch_effect_path, "train_epoch_effect.json")
    args.test_epoch_effect_path = os.path.join(epoch_effect_path, "test_epoch_effect.json")

    trainer = Trainer(
        gpus=args.device_ids,
        max_epochs=args.epochs,
        strategy="ddp",  # ddp_find_unused_parameters_false
        # checkpoint_callback=True,  # 将被移除用下面这个
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        logger=logger,
        sync_batchnorm=True,
        callbacks=[train_checkpoint_callback, test_checkpoint_callback],
        gradient_clip_val=0.5,  # 加入梯度裁剪
        resume_from_checkpoint=args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None,
    )

    trainer.fit(model, db)  # ckpt_path=resume_from_checkpoint

    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train")
    parser.add_argument("--model_select", default="bio_qwen2_5_vqa_model",
                        choices=[
                            "bio_qwen2_5_vqa_model", "bio_qwen2_5_vqa_model2", "bio_qwen2_5_vqa_model3"
                        ])
    parser.add_argument("--model_size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--load_ws_answer", default=True, choices=[True, False])
    parser.add_argument("--select_data", default="slake", choices=["slake", "rad", "path_vqa", "ovqa"])
    parser.add_argument("--select_mix", default="img_cutmix", choices=["img_mixup", "img_cutmix"])
    parser.add_argument("--mix_probability", default=0, type=float)
    parser.add_argument("--mix_alpha_1", default=5, type=int)
    parser.add_argument("--mix_alpha_2", default=1, type=int)
    parser.add_argument("--version", default="load_no_mix_no_aug")

    # configure
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device_ids", default=[0, 1, 2, 3])
    parser.add_argument("--num_workers", default=4, type=int)

    # model
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument("--weights_decay", default=0.05, type=float)
    parser.add_argument("--random_seed", default=1024, type=int)

    # constant_image
    parser.add_argument("--img_rotation", default=15, type=int)
    parser.add_argument("--resized_crop_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_right", default=1.0, type=float)
    parser.add_argument("--blur", default=[0.1, 2.0])
    parser.add_argument("--b_size", default=[5, 5])
    parser.add_argument("--blur_p", default=0.5, type=float)
    parser.add_argument("--apply_p", default=0.8, type=float)
    parser.add_argument("--img_flip", default=0.5, type=float)
    parser.add_argument("--brightness", default=0.4, type=float)
    parser.add_argument("--contrast", default=0.4, type=float)
    parser.add_argument("--saturation", default=0.4, type=float)
    parser.add_argument("--hue", default=0.4, type=float)
    parser.add_argument("--grayscale", default=0.2, type=float)

    # rand aug image
    parser.add_argument("--general_rand_aug", default=False)
    parser.add_argument("--resized_crop_scale_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_scale_right", default=1, type=float)
    parser.add_argument("--ra_n", default=2)
    parser.add_argument("--ra_m", default=12)
    parser.add_argument("--img_jitter", default=0.2, type=float)
    parser.add_argument("--reprob", default=0.2)

    # configure
    parser.add_argument("--epochs", default=15000, type=int)
    parser.add_argument("--qus_seq_len", default=30, type=int)
    parser.add_argument("--answer_open", default=0, type=int)
    parser.add_argument("--answer_close", default=1, type=int)
    parser.add_argument("--train_epoch_effect_path", default="param")
    parser.add_argument("--test_epoch_effect_path", default="param")

    parser.add_argument("--best_model_path", default="best_model")
    parser.add_argument("--test_best_model_path", default="test_best_model")
    parser.add_argument("--default_root_dir", default="./save/")
    parser.add_argument("--resume_from_checkpoint", default="./save/model/best_model/last.ckpt")
    parser.add_argument("--pre_best_model_path", default="./save/model/pre_best_model/0/train_loss=0.2651.ckpt")

    # slake dataset
    parser.add_argument("--slake_ans_ws_path", default="./save/ws/slake_ans_ws.pkl")
    parser.add_argument("--slake_ans_word_size", default=222, type=int)
    parser.add_argument("--slake_train_dataset_path", default="./data/ref/Slake1.0/train.json")
    parser.add_argument("--slake_test_dataset_path", default="./data/ref/Slake1.0/test.json")
    parser.add_argument("--slake_dataset_xm_path", default="./data/ref/Slake1.0/imgs/xmlab")

    # rad dataset
    parser.add_argument("--rad_ans_word_size", default=475, type=int)
    parser.add_argument("--rad_ans_ws_path", default="./save/ws/rad_ans_ws.pkl")
    parser.add_argument("--rad_images_path", default="./data/ref/rad/images")
    parser.add_argument("--rad_train_dataset_path", default="./data/ref/rad/trainset.json")
    parser.add_argument("--rad_test_dataset_path", default="./data/ref/rad/testset.json")

    # path_vqa dataset
    parser.add_argument("--path_vqa_ans_word_size", default=4092, type=int)
    parser.add_argument("--path_vqa_ans_ws_path", default="./save/ws/path_vqa_ans_ws.pkl")
    parser.add_argument("--path_train_img_folder_path", default="./data/ref/PathVQA/images/train")
    parser.add_argument("--path_test_img_folder_path", default="./data/ref/PathVQA/images/test")
    parser.add_argument("--path_train_dataset_text_path", default="./data/ref/PathVQA/qas/train/train_qa.pkl")
    parser.add_argument("--path_test_dataset_text_path", default="./data/ref/PathVQA/qas/test/test_qa.pkl")

    # ovqa dataset
    parser.add_argument("--ovqa_ans_word_size", default=707, type=int)
    parser.add_argument("--ovqa_ans_ws_path", default="./save/ws/ovqa_ans_ws.pkl")
    parser.add_argument("--ovqa_images_path", default="./data/ref/OVQA_publish/img")
    parser.add_argument("--ovqa_train_dataset_path", default="./data/ref/OVQA_publish/trainset.json")
    parser.add_argument("--ovqa_test_dataset_path", default="./data/ref/OVQA_publish/testset.json")

    # image
    parser.add_argument("--img_height", default=224, type=int)
    parser.add_argument("--img_width", default=224, type=int)

    args = parser.parse_args()

    main(args)


