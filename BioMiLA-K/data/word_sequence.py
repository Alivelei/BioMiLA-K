# _*_ coding: utf-8 _*_

"""
    @Time : 2023/1/23 13:55 
    @Author : smile 笑
    @File : word_sequence.py
    @desc :
"""


import pickle
import pandas as pd
import re
import json


def sentence_to_word(sentence, qus=True):
    if qus:
        queries = str(sentence).lower().strip("?").strip(" ").split(" ")  # 将问题进行切分，且都转换为小写
    else:
        queries = str(sentence).lower().strip(" ").strip(".")  # 将答案都转换为小写
    return queries


def pre_sentence_to_word(sentence):
    sentence = str(sentence).lower()
    queries = re.findall("[a-zA-Z]+", sentence)

    return queries


def word_id_transform(word_id_dict, sentence, max_len=None, pad_tag="PAD", unk=1):
    if max_len is not None:
        if max_len > len(sentence):
            sentence = sentence + [pad_tag] * (max_len - len(sentence))
        if max_len < len(sentence):
            sentence = sentence[:max_len]

    return [word_id_dict.get(word, unk) for word in sentence]


def id_word_transform(id_word_dict, indices):
    return [id_word_dict.get(idx) for idx in indices]


class Word2Sequence(object):
    PAD_TAG = "PAD"
    UNK_TAG = "UNK"

    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK
        }

        self.count = {}  # 统计词频

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_voc=None, max_voc=None, max_features=None):
        if min_voc is not None:
            self.count = {word: value for word, value in self.count.items() if value > min_voc}
        if max_voc is not None:
            self.count = {word: value for word, value in self.count.items() if value < max_voc}

        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def __len__(self):
        return len(self.dict)


class AnsWord2Sequence(object):
    def __init__(self):
        self.dict = {"UNK": 0}

        self.count = {}  # 统计词频

    def fit(self, word):
        self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self):
        for word in self.count:
            self.dict[word] = len(self.dict)

    def __len__(self):
        return len(self.dict)


def all_save_word2vec(csv_path, text_ws):
    captions = pd.read_csv(csv_path)["caption"]
    slake_queries = json.load(open("./ref/Slake1.0/train.json", encoding="utf-8"))
    rad_queries = json.load(open("./ref/rad/trainset.json", encoding="utf-8"))
    pathvqa_queries = pickle.load(open("./ref/PathVQA/qas/train/train_qa.pkl", "rb"))
    clef_queries = json.load(open("./ref/CLEFdata/train/train.json"))

    for caption in captions:
        text = pre_sentence_to_word(caption)  # 少于2个的不保存
        text_ws.fit(text)

    for query in slake_queries:
        if query["q_lang"] == "en":
            text = sentence_to_word(query["question"])
            text_ws.fit(text)

    for query in rad_queries:
        text = sentence_to_word(query["question"])
        text_ws.fit(text)

    for query in pathvqa_queries:
        text = sentence_to_word(query["question"])
        text_ws.fit(text)

    for query in clef_queries:
        text = sentence_to_word(query["question"])
        text_ws.fit(text)

    text_ws.build_vocab(min_voc=2)

    word_id_dict = text_ws.dict
    print(len(word_id_dict))
    pickle.dump(word_id_dict, open("../save/ws/all_word_id.pkl", "wb"))  # len 14745


def save_word2vec(queries, qus_ws, save_qus_path):
    for query in queries:
        # if query["q_lang"] == "en":
        text = sentence_to_word(query["question"])
        qus_ws.fit(text)
    qus_ws.build_vocab()

    word_id_dict = qus_ws.dict

    # print(word_id_dict)
    print(len(word_id_dict))
    pickle.dump(word_id_dict, open(save_qus_path, "wb"))


def ans_save_word2vec(queries, ans_ws, save_ans_path):
    for query in queries:
        # if query["q_lang"] == "en":
        # print(query["answer"])
        # text = sentence_to_word(query["answers"][0], False)
        text = sentence_to_word(query["Answer"], False)
        ans_ws.fit(text)
    ans_ws.build_vocab()

    word_id_dict = ans_ws.dict

    print(word_id_dict)
    print(len(word_id_dict))
    pickle.dump(word_id_dict, open(save_ans_path, "wb"))


def roco_compute_text_len(csv_path):
    csv = pd.read_csv(csv_path)

    text_len_list = []
    for line in csv["caption"]:
        text = pre_sentence_to_word(line)
        print(text)
        text_len_list.append(len(text))

    print(text_len_list)
    print(sum(text_len_list) // len(text_len_list))  # roco的caption长度平均值为20


if __name__ == '__main__':
    # all_save_word2vec("./ref/roco/train/radiologytraindata.csv", Word2Sequence())
    # roco_compute_text_len("./ref/roco/train/radiologytraindata.csv")

    # slake_qus_ws = Word2Sequence()
    # slake_ans_ws = AnsWord2Sequence()
    # slake_queries = json.load(open("./ref/Slake1.0/train.json", encoding="utf-8"))
    # rad_qus_ws = Word2Sequence()
    # rad_ans_ws = AnsWord2Sequence()
    # rad_queries = json.load(open("./ref/rad/trainset.json", encoding="utf-8"))
    # clef_qus_ws = Word2Sequence()
    # clef_ans_ws = AnsWord2Sequence()
    # clef_queries = json.load(open("./ref/CLEFdata/train/train.json"))
    #
    # # ans_save_word2vec(slake_queries, slake_ans_ws, "../save/ws/slake_ans_ws.pkl")  # 222
    # # ans_save_word2vec(rad_queries, rad_ans_ws, "../save/ws/rad_ans_ws.pkl")  # 475
    # ans_save_word2vec(clef_queries, clef_ans_ws, "../save/ws/clef_ans_ws.pkl")   # 1548

    # save_word2vec(slake_queries, slake_qus_ws, "../save/ws/slake_qus_ws.pkl")  # 305
    # save_word2vec(rad_queries, rad_qus_ws, "../save/ws/rad_qus_ws.pkl")  # 1231
    # save_word2vec(clef_queries, clef_qus_ws, "../save/ws/clef_qus_ws.pkl")  # 104

    # ovqa_qus_ws = Word2Sequence()
    # ovqa_ans_ws = AnsWord2Sequence()
    # ovqa_queries = json.load(open("./ref/OVQA_publish/trainset.json", encoding="utf-8"))
    #
    # save_word2vec(ovqa_queries, ovqa_qus_ws, "../save/ws/ovqa_qus_ws.pkl")  # 969
    # ans_save_word2vec(ovqa_queries, ovqa_ans_ws, "../save/ws/ovqa_ans_ws.pkl")  # 706

    # doc_vqa_ans_ws = AnsWord2Sequence()
    # doc_vqa_queries = json.load(open("./ref/DocVQA/train/train_v1.0.json", encoding="utf-8"))["data"]
    # ans_save_word2vec(doc_vqa_queries, doc_vqa_ans_ws, "../save/ws/doc_vqa_ans_ws.pkl")  # 25647

    earth_vqa_ans_ws = AnsWord2Sequence()
    earth_vqa_queries = json.load(open("./ref/EarthVQA/train.json", encoding="utf-8"))
    ans_save_word2vec(earth_vqa_queries, earth_vqa_ans_ws, "../save/ws/earth_vqa_ans_ws.pkl")  # 25647


