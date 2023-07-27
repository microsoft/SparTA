import argparse
import datasets
import numpy as np
import torch
import time
from nvitop import Device
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from transformers.models.opt.modeling_opt_triton import OPTModel

def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    memory_used = sum([device.memory_used() for device in devices])
    return memory_used / 1024 ** 3

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--local_model_path", type=str, help="Name path", default=None)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float32")
    parser.add_argument("--device_map", type=str, default="balanced", help="float16 or int8")

    return parser.parse_args()

def main():
    def get_time(cost, convert_time):
        t = config.num_hidden_layers / 20
        return cost * t, convert_time * t
    args = get_args()
    model_name = args.name
    local_model_path = args.local_model_path

    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
    )

    model = OPTModel.from_pretrained(
        model_name if not local_model_path else local_model_path,
        device_map=args.device_map,
        torch_dtype=torch.float32,
        pad_token_id=tokenizer.pad_token_id,
        offload_folder="/tmp/offload",
        # max_memory={0: '20GB', 1: '28GB', 2: '28GB', 3: '28GB', 4: '28GB', 5: '28GB', 6: '28GB', 7: '28GB', 'cpu': '400GB'},
        offload_state_dict=True,
    )
    model.eval()

    d = datasets.load_dataset("tatsu-lab/alpaca")["train"]
    N = len(d)
    bsz = 32
    test_time = 20
    seed = 171
    max_seq_length = 128
    device = "cuda"

    np.random.seed(seed)
    datas = []
    for _ in range(100):
        idx = np.random.randint(0, N - bsz)
        inputs = [d[ii + idx]["instruction"] + ' ' + d[ii + idx]["input"] + ' ' + d[ii + idx]["output"] for ii in range(bsz)]

        inputs = tokenizer(inputs, padding="max_length", max_length=max_seq_length, truncation=True)
        inputs = {ii: torch.tensor(jj).to(device) for ii, jj in inputs.items()}
        extended_attention_mask = model.get_extended_attention_mask(inputs["attention_mask"], inputs["input_ids"].size())
        base = ((extended_attention_mask == 0).sum(0).sum(0) > 0)
        x = base.repeat(extended_attention_mask.size(-1), 1)
        y = base.squeeze(0).unsqueeze(-1).repeat(1, extended_attention_mask.size(-1))
        extended_attention_mask = (x & y).unsqueeze(0).repeat(config.num_attention_heads, 1, 1).to(torch.int32)

        datas.append([inputs, extended_attention_mask])

    N = len(datas)
    random_idx = np.random.choice(range(N), test_time)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.time()
    for ii, idx in enumerate(random_idx):
        with torch.no_grad():
            model.decoder.layers[0].self_attn.spa.set_global_mask(datas[idx][1], True, 32, 32, config.num_attention_heads, device_num=8)
            model(**datas[idx][0])
        if ii == test_time // 2:
            memory = get_gpu_info()
            print("Memory", memory)
    torch.cuda.synchronize()
    end = time.time()
    cost = end - st
    convert_time = sum(model.decoder.layers[0].self_attn.spa.global_convert_overhead)
    cost, convert_time = get_time(cost, convert_time)
    print("Forward Implementation", cost)
    print("Convert Time", convert_time, "Convert Ratio", (convert_time / 1000) / cost)
    with open("results.txt", "a") as f:
        f.write(f"{model_name},PyTorch-S,{cost},{memory},{convert_time / 1000}\n")

if __name__ == "__main__":
    main()