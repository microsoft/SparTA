import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
)
import torch
import os
import joblib
from longformer.longformer_triton import Longformer, LongformerConfig, LongformerSelfAttention
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
import datasets
import numpy as np

from nvitop import Device
from sparta.opset.triton_dynamic_sparse_attention import TritonDynamicAttention
import argparse


def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    memory_used = sum([device.memory_used() for device in devices[:1]])
    return memory_used / 1024 ** 3

parser = argparse.ArgumentParser(description="Basic")
parser.add_argument("--model_name", type=str, default="longformer-base-4096/")
parser.add_argument("--max_seq_length", type=int, default=2048)
args = parser.parse_args()
model_name = args.model_name

device = torch.device('cuda:0')
config = LongformerConfig.from_pretrained(model_name)
config.attention_mode = 'sliding_chunks'
# config.attention_mode = 'n2'

model = Longformer.from_pretrained(model_name, config=config).to(device)
model = model.eval()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = model.config.max_position_embeddings

test_time = 100
seed = 171
max_seq_length = args.max_seq_length
bsz = 1
B, T, H, W = bsz, max_seq_length, 768, 256

data_dir = "./"
dataset_name = "longforomer_arxiv.pkl"
idx = list(range(1000))
np.random.shuffle(idx)
datas = joblib.load(os.path.join(data_dir, dataset_name))
datas = [{jj: torch.tensor(kk[ii:ii+1])[...,:T].cuda() for jj, kk in datas.items()} for ii in idx[:100]]

for ii in datas:
    ii["attention_mask"][ii["input_ids"] == 0] = 2
    ii["attention_mask"][ii["input_ids"] == 2] = 2
    # dynamic = (ii["input_ids"][0] == 2).nonzero()
    # dynamic = dynamic.squeeze().to(torch.int32)
    attention_mask = ii["attention_mask"]

    window_mask = torch.tril(torch.ones(T,T),diagonal=-W) + torch.triu(torch.ones(T,T),diagonal=W)
    window_mask = (window_mask == 0).int()
    window_mask = window_mask.unsqueeze(0).repeat(1, 1, 1).cuda()
    key_padding_mask = attention_mask == 0
    extra_attention_mask = attention_mask == 2
    remove_from_windowed_attention_mask = attention_mask != 1
    # global_row_mask = extra_attention_mask.unsqueeze(-1).repeat(1,1,T)
    global_col_mask = extra_attention_mask.unsqueeze(-2).repeat(1,T,1)
    key_padding_mask = key_padding_mask.unsqueeze(-2).repeat(1,T,1)
    mask = global_col_mask | window_mask
    mask = mask & ~key_padding_mask
    mask[ii["input_ids"] == 0] = 1
    # datas["attention_mask"][ii] = mask.squeeze(0)
    ii["attention_mask"] = mask
    # ii["dynamic"] = dynamicD

torch.cuda.empty_cache()

N = len(datas)
random_idx = np.random.choice(range(N), test_time)

torch.cuda.synchronize()
st = time.time()
for ii, idx in enumerate(random_idx):
    model(**datas[idx])
    if ii == test_time // 2:
        memory = get_gpu_info()
        print("Memory", memory)
torch.cuda.synchronize()
end = time.time()
print("Forward Implementation", end - st)

convert_time = sum(model.base_model.spa.global_convert_overhead)
a = 0
for k, v in model.named_modules():
    if isinstance(v, LongformerSelfAttention):
        a += v.time
print("Convert Time", convert_time, "Convert Ratio", (convert_time / 1000) / (end - st), a)

with open("results.txt", "a") as f:
    f.write(f"{model_name}_{max_seq_length},PyTorch-S,{end - st},{memory},{convert_time/1000}\n")

