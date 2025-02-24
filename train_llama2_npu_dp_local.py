import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP

from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer

from monitor import ColossalAIMonitor

def main():
    # --- 初始化分布式进程组 ---
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"npu:{local_rank}" if torch_npu.npu.is_available() else "cpu")
    torch_npu.npu.set_device(device)  # Ascend NPU 设置

    print(f"[INFO] rank={rank}/{world_size}, local_rank={local_rank}, device={device}")

    model_dir = snapshot_download(
        "modelscope/Llama-2-7b-chat-ms", 
        revision='v1.0.5'
    )
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)

    # 把模型加载到指定 device
    modelscope_model = Model.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=None  # 不要让 ModelScope 自动分片，自己DDP
    ).to(device)

    ddp_model = DDP(modelscope_model, device_ids=[local_rank], output_device=local_rank)
    
    # --- 初始化监控 ---
    monitor = ColossalAIMonitor(config_path="monitor_config.json")
    # 注册 Hook
    monitor.monitor(ddp_model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, 32000, (2, 16), dtype=torch.long, device=device)

    labels = input_ids.clone()

    # --- 训练循环 ---
    steps = 5
    for step in range(1, steps+1):
        optimizer.zero_grad()
        
        out = ddp_model(input_ids, labels=labels)
        loss = out.loss if hasattr(out, 'loss') else out[0].mean()
        
        loss.backward()
        optimizer.step()

        monitor.step_end(ddp_model)
        
        if rank == 0:
            print(f"[Step {step}/{steps}] loss={loss.item():.4f}")

    monitor.stop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
