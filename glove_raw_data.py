import os

from argparse import Namespace
from collections import Counter
import nltk.data
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
from konlpy.tag import Okt

# Global vars
MASK_TOKEN = "<MASK>"

args = Namespace(
    raw_dataset_txt="raw_data/NIRW2200000001.txt",
    window_size=2,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="raw_data/500_fasttext.csv",
    seed=1337,
)

# Split the raw text book into sentences
with open(args.raw_dataset_txt, "r", encoding="utf-8") as file:
    sentences = file.readlines()
sentences = sentences[:500]

okt = Okt()
tokenized_list = list()
vocab = {}
vocab_idx = -1
for i in tqdm(range(len(sentences))):
    if sentences[i][0] == " ":
        sentences[i] = sentences[i][1:]
    tokenized_document = okt.pos(sentences[i])
    temp = list()
    for token in tokenized_document:
        if token[1].lower() not in ["josa", "punctuation", "foreign"]:
            temp.append(token[0])
            try:
                vocab[token[0]]
            except KeyError:
                vocab[token[0]] = vocab_idx + 1
                vocab_idx += 1
    tokenized_list.append(" ".join(temp))

with open("raw_data/glove_vocab.json", "w") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

print(len(sentences), "sentences")
print("Sample:", sentences[100])

# Convert to dataframe
cbow_data = pd.DataFrame(tokenized_list, columns=["sentence"])

# Create split data
n = len(cbow_data)


def get_split(row_num):
    if row_num <= n * args.train_proportion:
        return "train"
    elif (row_num > n * args.train_proportion) and (row_num <= n * args.train_proportion + n * args.val_proportion):
        return "val"
    else:
        return "test"


cbow_data["split"] = cbow_data.apply(lambda row: get_split(row.name), axis=1)

print(cbow_data.head())


def calculate_cooccurrence_matrix(corpus, vocab, window_size=2):
    """
    주어진 코퍼스와 어휘 목록을 바탕으로 동시등장 확률 행렬을 계산합니다.

    :param corpus: 코퍼스 (단어의 리스트를 포함하는 리스트)
    :param vocab: 어휘 목록 (단어를 키로 하고 인덱스를 값으로 하는 딕셔너리)
    :param window_size: 문맥의 윈도우 크기
    :return: 동시등장 행렬 (numpy 배열)
    """
    vocab_size = len(vocab)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for sentence in tqdm(corpus, desc="calcul_comat"):
        token_list = sentence.split()
        sentence_length = len(token_list)
        for i, word in enumerate(token_list):
            if word in vocab:
                word_idx = vocab[word]
                start = max(0, i - window_size)
                end = min(sentence_length, i + window_size + 1)

                for j in range(start, end):
                    if i != j and token_list[j] in vocab:
                        context_word_idx = vocab[token_list[j]]
                        cooccurrence_matrix[word_idx, context_word_idx] += 1.0
                        cooccurrence_matrix[context_word_idx, word_idx] += 1.0

    # 동시등장 횟수를 확률로 변환
    cooccurrence_sum = np.sum(cooccurrence_matrix, axis=1, keepdims=True)
    cooccurrence_sum[cooccurrence_sum == 0] = 1  # 분모가 0인 경우를 1로 설정하여 나눗셈 오류 방지
    cooccurrence_matrix /= cooccurrence_sum

    return cooccurrence_matrix


train_comat = calculate_cooccurrence_matrix(
    cbow_data[cbow_data["split"] == "train"]["sentence"].tolist(), vocab, args.window_size
)
eval_comat = calculate_cooccurrence_matrix(
    cbow_data[cbow_data["split"] == "val"]["sentence"].tolist(), vocab, args.window_size
)
test_comat = calculate_cooccurrence_matrix(
    cbow_data[cbow_data["split"] == "test"]["sentence"].tolist(), vocab, args.window_size
)


np.savetxt("./raw_data/train_comat.npy", train_comat)
np.savetxt("./raw_data/eval_comat.npy", eval_comat)
np.savetxt("./raw_data/test_comat.npy", test_comat)
