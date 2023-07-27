import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer, # v4.25.1
)
from transformers.models.switch_transformers.modeling_switch_transformers_deepspeed import SwitchTransformersSparseMLP, SwitchTransformersModel
import torch
import os
import joblib
import numpy as np
import argparse
import datasets
import deepspeed

from nvitop import Device
from deepspeed.comm import init_distributed
init_distributed()

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

# Load to different GPU
# model.encoder.embed_tokens.to("cuda:0")
# for ii in range(6):
#     model.encoder.block[ii].to("cuda:0")
# for ii in range(6, 12):
#     model.encoder.block[ii].to("cuda:1")
# model.encoder.final_layer_norm.to("cuda:1")
# # model.decoder.embed_tokens.to("cuda:2")
# for ii in range(6):
#     model.decoder.block[ii].to("cuda:2")
# for ii in range(6, 12):
#     model.decoder.block[ii].to("cuda:3")
# model.decoder.final_layer_norm.to("cuda:3")

# parser = argparse.ArgumentParser(description="Basic")
# parser.add_argument("--dataset_name", type=str, default="")
# args = parser.parse_args()

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

for k, v in model.named_modules():
    if isinstance(v, SwitchTransformersSparseMLP):
        # print(k)
        v.fuse_expert(args.use_fp16 == "True")
model.eval()

model = deepspeed.init_inference(model,
                                dtype=torch.float32 if args.use_fp16 == "False" else torch.float16,
                                mp_size=1,
                                replace_with_kernel_inject=False,
                                replace_method="auto",
                                max_tokens=128,
                                save_mp_checkpoint_path=None,
                                )

N = len(datas)
random_idx = np.random.choice(range(N), test_time)
torch.cuda.empty_cache()
torch.cuda.synchronize()
st = time.time()
for ii, idx in enumerate(random_idx):
    # idx = random_idx[0]
    # print([jj.device for ii, jj in datas[idx].items()])
    with torch.no_grad():
        model(**datas[idx])
    if ii == test_time // 2:
        memory = get_gpu_info()
        print("Memory", memory)
torch.cuda.synchronize()
end = time.time()
print("Forward Implementation", end - st)

with open("results.txt", "a") as f:
    f.write(f"{args.use_fp16},{args.expert_number},{args.batch_size},DeepSpeed,{end - st},{memory}\n")
