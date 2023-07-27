import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.bert.modeling_bert_triton import BertForSequenceClassification
import torch
import os
import joblib
import numpy as np
import argparse

from nvitop import Device
from sparta.opset.triton_dynamic_sparse_attention import TritonDynamicAttention


def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    memory_used = sum([device.memory_used() for device in devices[:1]])
    return memory_used / 1024 ** 3


DATASETS = ["mnli", "mrpc", "cola", "rte", "qqp", "sst2", "wnli", "qnli", "stsb"]
parser = argparse.ArgumentParser(description="Basic")
parser.add_argument("--data_dir", type=str, default="data/glue/")
args = parser.parse_args()
data_dir = args.data_dir
dataset_name = "glue" if "glue" in data_dir else "long"

bsz = 32 if dataset_name == "glue" else 4
test_time = 100
seed = 171
max_seq_length = 128 if dataset_name == "glue" else 2048
model_name = "bert-base-uncased"

device = torch.device("cuda:0")
config = AutoConfig.from_pretrained(model_name, max_position_embeddings=max(max_seq_length, 512))
norm_model = BertForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True).to(device)
norm_model.eval()


for dataset_name in os.listdir(data_dir):
    print(dataset_name)
    np.random.seed(seed)
    datas = joblib.load(os.path.join(data_dir, dataset_name))
    if "glue" not in data_dir:
        datas = {ii: torch.tensor(jj) for ii, jj in datas.items() if ii != "label"}
        datas = [{jj: kk[ii:ii+bsz,:max_seq_length] for jj, kk in datas.items()} for ii in range(len(datas["input_ids"]) - bsz + 1)]
        datas = [{jj: kk.cuda() for jj, kk in ii.items()} for ii in datas]
    else:
        datas = [{jj: kk.cuda() for jj, kk in ii.items() if jj != "labels"} for ii in datas]
    N = len(datas)
    random_idx = [np.random.choice(list(range(N))) for ii in range(test_time)]

    torch.cuda.synchronize()
    st = time.time()
    for ii, idx in enumerate(random_idx):
        norm_model(**datas[idx])
        if ii == test_time // 2:
            memory = get_gpu_info()
            print("Memory", memory)
    torch.cuda.synchronize()
    end = time.time()
    print("Forward Implementation", end - st)
    convert_time = sum(norm_model.bert.spa.global_convert_overhead)

    print("Convert Time", convert_time, "Convert Ratio", (convert_time / 1000) / (end - st))
    with open("results.txt", "a") as f:
        f.write(f"{dataset_name},PyTorch-S,{end - st},{memory},{convert_time / 1000}\n")
    TritonDynamicAttention.global_convert_overhead = []

