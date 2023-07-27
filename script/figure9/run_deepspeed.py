import argparse
import datasets
import numpy as np
import torch
import time
from nvitop import Device
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from transformers.models.opt.modeling_opt_deepspeed import OPTModel, OPTDecoderLayer
import copy
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.inference.config import DeepSpeedTPConfig
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig

world_size = int(os.getenv('WORLD_SIZE', '4'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
deepspeed.init_distributed("nccl")

tp_config=DeepSpeedTPConfig()
tp_config.tp_size=1
tp_config.enabled=False


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
    args = get_args()
    model_name = args.name
    local_model_path = args.local_model_path

    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
    )

    # with deepspeed.OnDevice(dtype=torch.float32, device="meta"):
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = OPTModel.from_pretrained(
        model_name if not local_model_path else local_model_path,
        device_map=args.device_map,
        torch_dtype=torch.float32,
        pad_token_id=tokenizer.pad_token_id,
        offload_folder="/tmp/offload",
        # max_memory={0: '20GB', 1: '28GB', 2: '28GB', 3: '28GB', 4: '28GB', 5: '28GB', 6: '28GB', 7: '28GB', 'cpu': '400GB'},
        offload_state_dict=True,
    )
    def module_inject(layer_obj, model, config, micro_batch_size, max_seq_length, seed, preln, fp16=False):
        for name, child in model.named_children():
            if isinstance(child, layer_obj) and int(name) <= 1:
                cuda_config = DeepSpeedTransformerConfig(batch_size=micro_batch_size,
                                                        # max_seq_length=max_seq_length,
                                                        hidden_size=config.hidden_size,
                                                        intermediate_size=config.ffn_dim,
                                                        heads=config.num_attention_heads,
                                                        attn_dropout_ratio=config.attention_dropout,
                                                        hidden_dropout_ratio=config.dropout,
                                                        num_hidden_layers=config.num_hidden_layers,
                                                        initializer_range=config.init_std,
                                                        local_rank=-1,
                                                        seed=seed,
                                                        fp16=fp16,
                                                        pre_layer_norm=preln,
                                                        normalize_invertible=True,
                                                        gelu_checkpoint=True,
                                                        adjust_init_range=True,
                                                        attn_dropout_checkpoint=True,
                                                        stochastic_mode=True,
                                                        return_tuple=True,
                                                        training=False)

                new_module = DeepSpeedTransformerLayer(cuda_config)

                # copy relevant state from child -> new module
                qw = child.self_attn.q_proj.weight
                qb = child.self_attn.q_proj.bias
                kw = child.self_attn.k_proj.weight
                kb = child.self_attn.k_proj.bias
                vw = child.self_attn.v_proj.weight
                vb = child.self_attn.v_proj.bias

                qkvw = torch.cat((qw, kw, vw), 0)
                qkvb = torch.cat((qb, kb, vb), 0)

                new_module.attn_qkvw.data = qkvw
                new_module.attn_qkvb.data = qkvb
                new_module.attn_ow.data = child.self_attn.out_proj.weight
                new_module.attn_ob.data = child.self_attn.out_proj.bias
                if preln:
                    attention_layerNorm = child.self_attn_layer_norm
                else:
                    attention_layerNorm = child.self_attn_layer_norm
                new_module.attn_nw.data = attention_layerNorm.weight
                new_module.attn_nb.data = attention_layerNorm.bias
                if preln:
                    intermediate_FF = child.fc1
                else:
                    intermediate_FF = child.fc1
                new_module.inter_w.data = intermediate_FF.weight
                new_module.inter_b.data = intermediate_FF.bias
                new_module.output_w.data = child.fc2.weight
                new_module.output_b.data = child.fc2.bias
                if preln:
                    transformer_LayerNorm = child.final_layer_norm
                else:
                    transformer_LayerNorm = child.final_layer_norm
                new_module.norm_w.data = transformer_LayerNorm.weight
                new_module.norm_b.data = transformer_LayerNorm.bias

                setattr(model, name, copy.deepcopy(new_module).to(transformer_LayerNorm.weight.device))
                del child
                torch.cuda.empty_cache()
            else:
                module_inject(layer_obj, child, config, micro_batch_size, max_seq_length, seed, preln, fp16)

        return model
    # model = deepspeed.init_inference(model,
    #                                 dtype=torch.float32,
    #                                 mp_size=world_size,
    #                                 replace_with_kernel_inject=True,
    #                                 replace_method="auto",
    #                                 max_tokens=128,
    #                                 save_mp_checkpoint_path=None,
    #                                 )
    model = module_inject(OPTDecoderLayer, model, config, 32, 128, 171, config.do_layer_norm_before, fp16=False)
    # model = module_inject(OPTDecoderLayer, model, config, 8, 128, 42, config.do_layer_norm_before, fp16=False)
    # Initialize our Trainer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # model.eval()
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
        lens = inputs["attention_mask"].sum(-1).int()
        datas.append([inputs, lens])

    N = len(datas)
    random_idx = np.random.choice(range(N), test_time)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.time()
    for ii, idx in enumerate(random_idx):
        with torch.no_grad():
            model(use_cache=False, **datas[idx][0])
        if ii == test_time // 2:
            memory = get_gpu_info()
            print("Memory", memory)
    torch.cuda.synchronize()
    end = time.time()
    print("Forward Implementation", end - st)
    with open("results.txt", "a") as f:
        f.write(f"{model_name},DeepSpeed,{end - st},{memory}\n")

if __name__ == "__main__":
    main()