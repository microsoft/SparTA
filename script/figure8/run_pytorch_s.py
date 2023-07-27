import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from transformers.models.switch_transformers.modeling_switch_transformers_triton import SwitchTransformersSparseMLP, SwitchTransformersModel
import torch
import os
import joblib
import numpy as np
import argparse
import datasets

from nvitop import Device

def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    memory_used = sum([device.memory_used() for device in devices])
    return memory_used / 1024 ** 3

test_time = 100
seed = 171
max_seq_length = 128
parser = argparse.ArgumentParser(description="Basic")
parser.add_argument("--expert_number", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--use_fp16", type=str, default="False")
args = parser.parse_args()
bsz = args.batch_size

model_name = f"google/switch-base-{args.expert_number}"
device = torch.device("cuda:0")
config = AutoConfig.from_pretrained(model_name, max_position_embeddings=max_seq_length)
model = SwitchTransformersModel.from_pretrained(model_name, config=config).cuda()
model.eval()
if args.use_fp16 == "True":
    model = model.half()

tokenizer = AutoTokenizer.from_pretrained(model_name)

d = datasets.load_dataset("glue", "mnli")
N = len(d["train"])

np.random.seed(seed)
datas = []
for ii in range(100):
    idx = np.random.randint(0, N)
    inputs = [d["train"][ii + idx]["premise"] + '</s>' + d["train"][ii + idx]["hypothesis"] for ii in range(bsz)]

    inputs = tokenizer(inputs, padding="max_length", max_length=max_seq_length, truncation=True)
    inputs = {ii: torch.tensor(jj).to(device) for ii, jj in inputs.items()}
    inputs["decoder_input_ids"] = inputs["input_ids"]
    datas.append(inputs)

N = len(datas)
random_idx = np.random.choice(range(N), test_time)

torch.cuda.empty_cache()
torch.cuda.synchronize()
st = time.time()
for ii, idx in enumerate(random_idx):
    # idx = random_idx[0]
    with torch.no_grad():
        model(**datas[idx])
    torch.cuda.empty_cache()
    if ii == test_time // 2:
        memory = get_gpu_info()
        print("Memory", memory)
torch.cuda.synchronize()
end = time.time()
print("Forward Implementation", end - st)
convert_time = sum(model.spa.global_convert_overhead) + sum(model.cross_spa.global_convert_overhead)
print("Convert Time", convert_time, "Convert Ratio", (convert_time / 1000) / (end - st))
with open("results.txt", "a") as f:
    f.write(f"{args.use_fp16},{args.expert_number},{args.batch_size},PyTorch-S,{end - st},{memory},{convert_time/1000}\n")
