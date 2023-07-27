import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.bert.modeling_bert_pit import BertForSequenceClassification
import torch
import os
import joblib
import numpy as np
from sparta.opset.seqlen_dynamic_sparse_linear import SeqlenDynamicSparseLinear
from nvitop import Device
import argparse


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

modules = {}
for k, v in norm_model.named_modules():
    modules[k] = v
    if isinstance(v, nn.Linear) and not k.startswith("bert.pooler") and not k.startswith("classifier"):
        parts = k.split(".")
        father_module_name = ".".join(parts[:-1])
        child_name = parts[-1]
        father = modules[father_module_name]
        setattr(father, child_name, SeqlenDynamicSparseLinear(v, True))
norm_model.eval()


for dataset_name in os.listdir(data_dir):
    if not dataset_name.endswith(".pkl"):
        continue
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

    torch.cuda.empty_cache()
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

    with open("results.txt", "a") as f:
        f.write(f"{dataset_name},PIT,{end - st},{memory}\n")

