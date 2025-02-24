import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_npu
except ImportError:
    torch_npu = None

from diffusers import StableDiffusionPipeline

from monitor import ColossalAIMonitor


def main():
    # --- 分布式初始化 ---
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # local_rank / device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Ascend NPU 场景
    if torch_npu and torch_npu.npu.is_available():
        device = torch.device(f"npu:{local_rank}")
        torch_npu.npu.set_device(device)
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[INFO] rank={rank}/{world_size}, local_rank={local_rank}, device={device}")

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16
    )
    pipe.to(device)

    unet_model = pipe.unet
    unet_model = unet_model.to(device)

    # --- 用DDP包装 unet ---
    ddp_unet = DDP(unet_model, device_ids=[local_rank], output_device=local_rank)

    # --- 初始化Monitor, 注册 Hook ---
    monitor = ColossalAIMonitor(config_path="monitor_config.json")
    monitor.monitor(ddp_unet)

    optimizer = optim.SGD(ddp_unet.parameters(), lr=1e-4)

    # --- 模拟一个最小训练循环 ---
    steps = 3
    batch_size = 2
    latent_channels = 4
    latent_height = 64
    latent_width  = 64

    for step in range(1, steps+1):
        optimizer.zero_grad()
    
        latents = torch.randn(
            batch_size, latent_channels, latent_height, latent_width,
            dtype=torch.float16,
            device=device
        )
        timesteps = torch.randint(
            0, 1000, (batch_size,),
            dtype=torch.long,
            device=device
        )
        text_embeds = torch.randn(
            batch_size, 8, 768,
            dtype=torch.float16,
            device=device
        )
    
        out = ddp_unet(
            latents, timesteps,
            encoder_hidden_states=text_embeds,
            return_dict=True
        )
        if isinstance(out, dict) and 'sample' in out:
            prediction = out['sample']
            loss = prediction.mean()
        else:
            # fallback
            loss = out[0].mean() if isinstance(out, (tuple, list)) else out.mean()

        loss.backward()
        optimizer.step()

        monitor.step_end(ddp_unet)

        if rank == 0:
            print(f"[Step {step}/{steps}] loss = {loss.item():.4f}")

    monitor.stop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
