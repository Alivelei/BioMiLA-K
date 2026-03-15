# _*_ coding: utf-8 _*_

"""
    @Time : 2024/9/20 21:01 
    @Author : smile 笑
    @File : datasets.py
    @desc :
"""


from torch.utils.data import Dataset
import re
import json
import pickle
import os
import torchvision.transforms as tfs
from PIL import Image
from data.word_sequence import sentence_to_word, word_id_transform
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_aug_img(img, args):
    if args.general_rand_aug:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_scale_left, args.resized_crop_scale_right),
                                  ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(p=args.img_flip),
            tfs.RandAugment(args.ra_n, args.ra_m),
            tfs.ColorJitter(args.img_jitter, args.img_jitter, args.img_jitter),
            tfs.ToTensor(),
            tfs.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_left, args.resized_crop_right)),
            tfs.RandomApply([tfs.GaussianBlur(kernel_size=args.b_size, sigma=args.blur)], p=args.blur_p),
            tfs.RandomGrayscale(p=args.grayscale),
            tfs.RandomApply([
                tfs.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                p=args.apply_p
            ),
            tfs.RandomRotation(args.img_rotation),
            tfs.RandomHorizontalFlip(args.img_flip),
            tfs.ToTensor(),
            tfs.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        # aug = tfs.Compose([
        #     tfs.Resize([args.img_height, args.img_width]),
        #     tfs.ToTensor(),
        #     tfs.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # ])

    return aug(img)  # 归一化图像


def test_aug_img(img, args):
    aug = tfs.Compose([
        tfs.Resize([args.img_height, args.img_width]),
        tfs.ToTensor(),
        tfs.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return aug(img)


class SlakeDatasetModule(Dataset):
    def __init__(self, args, dataset_path, mode):
        self.args = args

        self.mode = mode
        self.xm_path = args.slake_dataset_xm_path
        self.queries = json.load(open(dataset_path, encoding="utf-8"))

        self.queries = [query for query in self.queries if query["q_lang"] == "en"]

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.xm_path + str(query["img_id"]), "source.jpg")

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        return image, query["question"], query["answer"], img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class RadDatasetModule(Dataset):
    def __init__(self, args, rad_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.rad_images_path
        self.queries = json.load(open(rad_dataset_path, encoding="utf-8"))

    def __getitem__(self, idx):
        query = self.queries[idx]
        img_path = os.path.join(self.images_path, str(query["image_name"]))

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        return image, query["question"], query["answer"], img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class PathVQADatasetModule(Dataset):
    def __init__(self, args, img_folder_path, dataset_text_path, mode):
        self.args = args
        self.mode = mode
        self.img_folder_path = img_folder_path

        self.queries = pickle.load(open(dataset_text_path, "rb"))

    def word_2id(self, question, dic, max_seq_len=20):
        sentence = [i for i in re.findall("[a-z0-9]*", question.lower()) if len(i) > 0]

        if max_seq_len is not None:
            if max_seq_len > len(sentence):
                sentence = sentence + ["<unk>"] * (max_seq_len - len(sentence))
            if max_seq_len < len(question):
                sentence = sentence[:max_seq_len]
        return [int(dic.get(word, dic["<unk>"])) for word in sentence]

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.img_folder_path, query["image"] + ".jpg")

        answer = query["answer"]

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        if answer.lower() == "no" or answer.lower() == "yes":
            answer_type_id = self.args.answer_close
        else:
            answer_type_id = self.args.answer_open

        return image, query["question"], str(query["answer"]), img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class OVQADatasetModule(Dataset):
    def __init__(self, args, ovqa_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.ovqa_images_path
        self.queries = json.load(open(ovqa_dataset_path, encoding="utf-8"))

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.images_path, str(query["image_name"]))

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        return image, query["question"], str(query["answer"]), img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class LoadWsSlakeDatasetModule(Dataset):
    def __init__(self, args, dataset_path, mode):
        self.args = args

        self.mode = mode
        self.xm_path = args.slake_dataset_xm_path
        self.queries = json.load(open(dataset_path, encoding="utf-8"))

        self.queries = [query for query in self.queries if query["q_lang"] == "en"]

        self.ans_ws = pickle.load(open(args.slake_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.xm_path + str(query["img_id"]), "source.jpg")

        # question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(query["answer"], False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, query["question"], ans_id, img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class LoadWsRadDatasetModule(Dataset):
    def __init__(self, args, rad_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.rad_images_path
        self.queries = json.load(open(rad_dataset_path, encoding="utf-8"))

        self.ans_ws = pickle.load(open(args.rad_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

    def __getitem__(self, idx):
        query = self.queries[idx]
        img_path = os.path.join(self.images_path, str(query["image_name"]))

        # question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, query["question"], ans_id, img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class LoadWsPathVQADatasetModule(Dataset):
    def __init__(self, args, img_folder_path, dataset_text_path, mode):
        self.args = args
        self.mode = mode
        self.img_folder_path = img_folder_path

        self.queries = pickle.load(open(dataset_text_path, "rb"))

        self.max_seq_len = args.qus_seq_len
        self.ans_ws = pickle.load(open(args.path_vqa_ans_ws_path, "rb"))

    def word_2id(self, question, dic, max_seq_len=20):
        sentence = [i for i in re.findall("[a-z0-9]*", question.lower()) if len(i) > 0]

        if max_seq_len is not None:
            if max_seq_len > len(sentence):
                sentence = sentence + ["<unk>"] * (max_seq_len - len(sentence))
            if max_seq_len < len(question):
                sentence = sentence[:max_seq_len]
        return [int(dic.get(word, dic["<unk>"])) for word in sentence]

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.img_folder_path, query["image"] + ".jpg")

        answer = query["answer"]

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        if answer.lower() == "no" or answer.lower() == "yes":
            answer_type_id = self.args.answer_close
        else:
            answer_type_id = self.args.answer_open

        # question = sentence_to_word(query["question"])

        ans_id = self.ans_ws.get(answer, self.ans_ws["unknown"])

        return image, query["question"], ans_id, img_path, answer_type_id

    def __len__(self):
        return len(self.queries)


class LoadWsOVQADatasetModule(Dataset):
    def __init__(self, args, ovqa_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.ovqa_images_path
        self.queries = json.load(open(ovqa_dataset_path, encoding="utf-8"))

        self.ans_ws = pickle.load(open(args.ovqa_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.images_path, str(query["image_name"]))

        # question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, query["question"], ans_id, img_path, answer_type_id

    def __len__(self):
        return len(self.queries)

