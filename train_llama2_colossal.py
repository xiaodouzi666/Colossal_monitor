import os
import torch
import torch.optim as optim

# 如果你使用Ascend NPU，需要保留 torch_npu
import torch_npu

# ColossalAI 导入
import colossalai
from colossalai.launch import launch
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer import ZeroOptimizer

# modelscope llama2
from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer

# 从你的最新 monitor.py 中导入
from monitor import ColossalAIMonitor


def main():
    """
    用 ColossalAI + ZeRO3 并行来训练 LLaMA2，并使用 ColossalAIMonitor 来采集梯度信息
    """

    #--------------------------------------------------------------------------
    # 1) 准备 ColossalAI 的分布式并行配置
    #--------------------------------------------------------------------------
    # 你可根据需要修改 parallel/tensor/pipeline/zero/fp16 等字段。
    launch_config = {
        "parallel": {
            "data": 1,        # 数据并行大小 (demo: 单纯依赖ZeRO)
            "tensor": 1,      # 张量并行大小
            "pipeline": 1     # 流水并行大小
        },
        "zero": {
            "level": 3        # 这里演示 ZeRO Stage-3
        },
        "fp16": {
            "mode": "amp",            # 自动混合精度
            "auto_loss_scale": True
        }
    }

    #--------------------------------------------------------------------------
    # 2) 从环境变量解析本地 rank, host, port
    #--------------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))

    #--------------------------------------------------------------------------
    # 3) 启动 ColossalAI
    #   这一步会自动初始化分布式进程组, 并根据配置创建通信组
    #--------------------------------------------------------------------------
    launch(
        config=launch_config,
        rank=local_rank,
        world_size=world_size,
        host=host,
        port=port,
        backend="hccl"   # 华为 Ascend 通常用 'hccl' 作为后端
    )

    #--------------------------------------------------------------------------
    # 4) Ascend NPU 设备 (可选)
    #   一般 ColossalAI 会自动分配device, 但如果你想手动设置，也可以:
    #--------------------------------------------------------------------------
    if torch_npu and torch_npu.npu.is_available():
        device = torch.device(f"npu:{local_rank}")
        torch_npu.npu.set_device(device)
    else:
        device = torch.device("cpu")

    rank = torch.distributed.get_rank()
    print(f"[INFO] rank={rank}/{world_size}, local_rank={local_rank}, device={device}")

    #--------------------------------------------------------------------------
    # 5) 下载 LLaMA2 模型 & tokenizer
    #--------------------------------------------------------------------------
    model_dir = snapshot_download(
        "modelscope/Llama-2-7b-chat-ms",
        revision='v1.0.5'
    )
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)

    #--------------------------------------------------------------------------
    # 6) 加载原模型到 device (fp16)
    #--------------------------------------------------------------------------
    modelscope_model = Model.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=None  # 不让 ModelScope 自行分片
    ).to(device)

    #--------------------------------------------------------------------------
    # 7) 用常规 PyTorch 优化器
    #--------------------------------------------------------------------------
    base_optimizer = optim.SGD(modelscope_model.parameters(), lr=1e-4)

    #--------------------------------------------------------------------------
    # 8) 构建 ColossalAI Zero 优化器 & ZeroDDP 包装
    #   - ZeroOptimizer会管理优化器状态分片
    #   - ZeroDDP会管理参数/梯度分片
    #--------------------------------------------------------------------------
    #  (a) 先用 ZeroDDP 包装模型
    ddp_model = ZeroDDP(
        module=modelscope_model,
        zero_stage=3
    )
    #  (b) 再用 ZeroOptimizer 包装基础优化器
    #     注意: 这里必须传入 ddp_model.parameters(), 不能是 modelscope_model.parameters()
    optimizer = ZeroOptimizer(
        optimizer=base_optimizer,
        model=ddp_model,
        zero_stage=3
    )

    #--------------------------------------------------------------------------
    # 9) 初始化监控
    #--------------------------------------------------------------------------
    monitor = ColossalAIMonitor(config_path="monitor_config.json")
    # 注册 Hook，获取本地梯度
    monitor.monitor(ddp_model)

    #--------------------------------------------------------------------------
    # 10) 一点随机输入 (demo)
    #--------------------------------------------------------------------------
    input_ids = torch.randint(0, 32000, (2, 16), dtype=torch.long, device=device)
    labels = input_ids.clone()

    #--------------------------------------------------------------------------
    # 11) 训练循环 (只跑几步作演示)
    #--------------------------------------------------------------------------
    steps = 5
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        out = ddp_model(input_ids, labels=labels)
        loss = out.loss if hasattr(out, 'loss') else out[0].mean()

        # => 反向传播 => 这里会触发 monitor 的 Hook 收集 local grad (分片)
        ddp_model.backward(loss)

        # => 更新 (ZeRO3 下会在内部完成 shard sync)
        optimizer.step()

        # => 调用 monitor.step_end() 收集 "全局" 统计
        monitor.step_end(ddp_model)

        if rank == 0:
            print(f"[Step {step}/{steps}] loss={loss.item():.4f}")

    #--------------------------------------------------------------------------
    # 12) 训练结束收尾
    #--------------------------------------------------------------------------
    monitor.stop()
    torch.distributed.barrier()
    # ColossalAI 通常不强制你调用 dist.destroy_process_group()，可酌情加
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
