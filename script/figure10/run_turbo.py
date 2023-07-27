import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    BertModel,
)
import torch
import os
import joblib
import numpy as np
import argparse

from nvitop import Device

import turbo_transformers


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
norm_model = BertModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True).to(device)
norm_model.eval()

# norm_model = turbo_transformers.BertModel.from_torch(norm_model, backend="turbo")
norm_model = turbo_transformers.BertModelSmartBatch.from_torch(norm_model)


for dataset_name in os.listdir(data_dir):
    def convert_bsz1(input_ids, attention_mask):
        return [ii[:jj.sum()].unsqueeze(0).cuda() for ii, jj in zip(input_ids, attention_mask)], [jj.sum().item() for ii, jj in zip(input_ids, attention_mask)]
    if not dataset_name.endswith(".pkl"):
        continue
    print(dataset_name)
    np.random.seed(seed)
    datas = joblib.load(os.path.join(data_dir, dataset_name))
    if "glue" not in data_dir:
        datas = {ii: torch.tensor(jj) for ii, jj in datas.items() if ii != "label"}
        datas = [{jj: kk[ii:ii+bsz,:max_seq_length] for jj, kk in datas.items()} for ii in range(len(datas["input_ids"]) - bsz + 1)]
        datas = [convert_bsz1(ii["input_ids"], ii["attention_mask"]) for ii in datas]
    else:
        datas = [convert_bsz1(ii["input_ids"], ii["attention_mask"]) for ii in datas]
    N = len(datas)
    random_idx = [np.random.choice(list(range(N))) for ii in range(test_time)]

    torch.cuda.synchronize()
    st = time.time()
    for ii, idx in enumerate(random_idx):
        norm_model(datas[idx][0], datas[idx][1])
        if ii == test_time // 2:
            memory = get_gpu_info()
            print("Memory", memory)
    torch.cuda.synchronize()
    end = time.time()
    print("Forward Implementation", end - st)

    with open("results.txt", "a") as f:
        f.write(f"{dataset_name},Turbo,{end - st},{memory}\n")


