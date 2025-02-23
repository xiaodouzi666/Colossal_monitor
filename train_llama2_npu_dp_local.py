import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# 如果是华为Ascend NPU需要的库
import torch_npu  # 若环境已安装
from torch.nn.parallel import DistributedDataParallel as DDP

from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer

# 导入你之前的 ColossalAIMonitor (确保 monitor.py 或同等代码在同目录/可import路径)
from monitor import ColossalAIMonitor

def main():
    # --- 1. 初始化分布式进程组 ---
    # 推荐在外部用: torchrun --nproc_per_node=2 python train_llama2_npu_dp_local.py
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl')  # 如在Ascend上多用HCCL；若是GPU可用nccl
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"npu:{local_rank}" if torch_npu.npu.is_available() else "cpu")
    torch_npu.npu.set_device(device)  # Ascend NPU 设置

    print(f"[INFO] rank={rank}/{world_size}, local_rank={local_rank}, device={device}")

    # --- 2. 下载 & 加载 LLaMA2 模型 (示例: 'modelscope/Llama-2-7b-chat-ms') ---
    # 仅下载meta信息和非 .bin 文件(权重)时可用 ignore_file_pattern=[r'.+\.bin$']，实际要拉权重需去掉
    model_dir = snapshot_download(
        "modelscope/Llama-2-7b-chat-ms", 
        revision='v1.0.5'
    )
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)

    # 把模型加载到指定 device
    # 注意: device_map 如果写{'': str(device)}，ModelScope 会尝试分片加载，这里演示单进程内放全部到 local_rank
    modelscope_model = Model.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=None  # 不要让 ModelScope 自动分片；我们自己DDP
    ).to(device)

    # --- 3. 包装 DDP 以启用数据并行 (DP) ---
    # 这里 modelscope_model 本质是一个 huggingface 类似的 nn.Module
    # DDP会在 backward() 做 AllReduce => 产生"聚合后"梯度
    ddp_model = DDP(modelscope_model, device_ids=[local_rank], output_device=local_rank)
    
    # --- 4. 初始化监控 (monitor_config.json) ---
    monitor = ColossalAIMonitor(config_path="monitor_config.json")
    # 注册 Hook，捕获“聚合前”(local)梯度
    monitor.monitor(ddp_model)

    # --- 5. 模拟数据 & 优化器 (仅演示 backward) ---
    # 真实微调需要实际的 input_ids, labels。这里用随机tensor模拟
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-4)
    
    # 构造一个随机输入/labels(假定 sequence长度=16)
    input_ids = torch.randint(0, 32000, (2, 16), dtype=torch.long, device=device)
    # llama2通常是因果LM，需要 labels=input_ids(shifted)或自定义
    # 这里只是演示 backward，不注重真实Loss
    labels = input_ids.clone()

    # --- 6. 训练循环 ---
    steps = 5
    for step in range(1, steps+1):
        optimizer.zero_grad()
        
        # (A) 这里你的 modelscope_model 可能需要 .forward(**kwargs)
        #     有些 LLaMA2 ModelScope会给出 model(input_ids, labels=...) => (loss, logits)
        out = ddp_model(input_ids, labels=labels)
        # out 可能是一个 namedtuple 或 dict, 里面含 .loss / .logits
        # 具体要打印查看 model.forward() 的返回
        loss = out.loss if hasattr(out, 'loss') else out[0].mean()
        
        loss.backward()  # 触发 monitor 的本地梯度 Hook
        optimizer.step()

        # 在 DP 下, ddp_model在 backward里已做AllReduce => 产生"聚合后"梯度
        monitor.step_end(ddp_model)  # 记录全局梯度
        
        if rank == 0:
            print(f"[Step {step}/{steps}] loss={loss.item():.4f}")

    # --- 7. 收尾 ---
    monitor.stop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
