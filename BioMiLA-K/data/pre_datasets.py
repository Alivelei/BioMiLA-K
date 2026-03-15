# _*_ coding: utf-8 _*_

"""
    @Time : 2024/9/20 21:16 
    @Author : smile 笑
    @File : pre_datasets.py
    @desc :
"""


from PIL import Image
import torchvision.transforms as tfs
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import json
import os
import random


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def aug_img(img, image_size):
    aug = tfs.Compose([
        tfs.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        tfs.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        tfs.ToTensor(),
        tfs.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return aug(img)


class PreCaptionROCOV2Dataset(Dataset):
    PROMPT_SENTENCES = [
        "Describe the image concisely.",
        "Please generate a caption based on this image.",
        "Create a description for the given picture.",
        "Provide a detailed caption for this photo.",
        "Can you describe this image in one sentence?",
        "Generate a descriptive caption for the displayed image.",
        "Offer a brief caption that fits this picture.",
        "Write a sentence summarizing what is shown in this image.",
        "Formulate a caption for the content of this image.",
        "Write a terse but informative summary of the picture.",
        "Give a caption that reflects the essence of this picture."
    ]

    def __init__(self, args):
        self.args = args

        self.merged_data = json.load(open(args.rocov2_merged_data_path))

    def __getitem__(self, item):
        query = self.merged_data[item]
        image_folder = "valid" if query["mode"] == "validation" else query["mode"]
        img_path = os.path.join(self.args.all_pretrained_data_path + image_folder, str(query["image_name"] + ".jpg"))
        image = aug_img(Image.open(img_path).convert("RGB"), self.args.image_size)
        text_input = np.random.choice(self.PROMPT_SENTENCES)
        caption = query["caption"]

        return image, text_input, caption, img_path

    def __len__(self):
        return len(self.merged_data)


class PreQADescriptionROCOv2Dataset(Dataset):
    def __init__(self, args):
        self.args = args

        # qa_json_data = json.load(open(args.rocov2_qa_clean_data))
        # description_json_data = json.load(open(args.rocov2_description_clean_data))
        # self.all_data = qa_json_data + description_json_data
        self.all_data = json.load(open(args.rocov2_filtered_instruct_data))

    def __getitem__(self, item):
        query = self.all_data[item]
        image_folder = "valid" if query["mode"] == "validation" else query["mode"]
        img_path = os.path.join(self.args.all_pretrained_data_path + image_folder, str(query["image_name"] + ".jpg"))
        image = aug_img(Image.open(img_path).convert("RGB"), self.args.image_size)
        text_input = query["question"]
        answer = query["answer"]

        return image, text_input, answer, img_path

    def __len__(self):
        return len(self.all_data)


class PreQALoRaROCOv2Dataset(Dataset):
    PROMPT_SENTENCES = [
        "Describe the following image in detail.",
        "Provide a detailed description of the given image.",
        "Give an elaborate explanation of the image you see.",
        "Share a comprehensive rundown of the presented image.",
        "Offer a thorough analysis of the image.",
        "Explain the various aspects of the image before you.",
        "Clarify the contents of the displayed image with great detail.",
        "Characterize the image using a well-detailed description.",
        "Break down the elements of the image in a detailed manner.",
        "Walk through the important details of the image.",
        "Portray the image with a rich, descriptive narrative.",
        "Narrate the contents of the image with precision.",
        "Analyze the image in a comprehensive and detailed manner.",
        "Illustrate the image through a descriptive explanation.",
        "Examine the image closely and share its details.",
        "Write an exhaustive depiction of the given image.",
    ]

    def __init__(self, args):
        self.args = args

        self.all_data = json.load(open(args.rocov2_filtered_instruct_data))

    def __getitem__(self, item):
        query = self.all_data[item]
        image_folder = query["mode"]
        img_path = os.path.join(self.args.all_pretrained_data_path + image_folder, str(query["image_name"] + ".jpg"))
        image = aug_img(Image.open(img_path).convert("RGB"), self.args.image_size)
        text_input = query.get("question", random.choice(self.PROMPT_SENTENCES))  # 对detailed数据则随机抽问题
        answer = query["answer"]

        return image, text_input, answer, img_path

    def __len__(self):
        return len(self.all_data)

