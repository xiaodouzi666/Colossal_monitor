import os
import json
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Ascend NPU 支持(若无此环境可注释掉)
try:
    import torch_npu
except ImportError:
    torch_npu = None

def distributed_is_initialized():
    """ 检查是否已经在分布式环境中（已初始化） """
    return dist.is_available() and dist.is_initialized()

def try_init_distributed_backend(backend='hccl'):
    # 这里就直接用 dist
    if dist.is_available() and not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        if world_size > 1:
            print(f"[INFO] Init process group: rank={rank}, world_size={world_size}, "
                  f"master={master_addr}:{master_port}, backend={backend}")
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )
        else:
            print("[INFO] world_size=1, no need to init_process_group()")


class ColossalAIMonitor:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.grad_records_local = []   # 暂存“聚合前”统计
        self.grad_records_global = []  # 暂存“聚合后”统计
        self.global_step = 0

        # 从配置文件中获取各种设置
        self.output_file_local  = self.config.get('output_file_local',  'grad_stats_local.csv')
        self.output_file_global = self.config.get('output_file_global', 'grad_stats_global.csv')
        self.param_list = self.config.get('param_list', [])   # 若为空则监控全部
        self.stats_to_calc = self.config.get('stats', ["norm", "mean", "max", "min"])
        self.write_interval = self.config.get('write_interval', 1)

        # 并行模式 (示例：dp / tp / pp / zero1 / zero2 / zero3 / none)
        self.parallel_mode = self.config.get('parallel_mode', 'dp')

        # 分布式信息
        if distributed_is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _load_config(self, path):
        """ 读取 JSON 配置文件，如果未提供则返回空配置。 """
        if path is None:
            print("[ColossalAIMonitor] No config file provided, using default settings.")
            return {}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, 'r') as f:
            conf = json.load(f)
        print(f"[ColossalAIMonitor] Loaded config from {path}")
        return conf

    def _create_hook_fn(self, param_name):
        """ 在“本地梯度”计算完时，记录一次统计(聚合前). """
        def hook_fn(grad):
            record = {
                'step': self.global_step,
                'rank': self.rank,
                'param_name': param_name,
            }
            self._calc_stats(record, grad, suffix="")
            self.grad_records_local.append(record)
        return hook_fn

    def _calc_stats(self, record_dict, grad_tensor, suffix=""):
        """ 根据 self.stats_to_calc 计算 norm/mean/max/min 并写入 record_dict. """
        if "norm" in self.stats_to_calc:
            record_dict["norm"+suffix] = float(grad_tensor.norm().item())
        if "mean" in self.stats_to_calc:
            record_dict["mean"+suffix] = float(grad_tensor.mean().item())
        if "max" in self.stats_to_calc:
            record_dict["max"+suffix] = float(grad_tensor.max().item())
        if "min" in self.stats_to_calc:
            record_dict["min"+suffix] = float(grad_tensor.min().item())

    def monitor(self, model):
        """ 用户在训练前调用此方法，为 model 注册梯度 Hook（采集本地梯度）。 """
        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.requires_grad:
                param.register_hook(self._create_hook_fn(name))
        print(f"[ColossalAIMonitor] Hook registration done. parallel_mode={self.parallel_mode}")

    def step_end(self, model):
        """
        每步训练完成后由用户显式调用。
        - 先增加 step 计数；
        - 再根据并行模式，对“聚合后”的梯度进行采集；
        - 若到达写文件间隔，就写到CSV。
        """
        self.global_step += 1

        # ------ 这里做“聚合后”统计 ------
        if self.parallel_mode == 'none':
            self._collect_global_stats_single(model)
        elif self.parallel_mode == 'dp':
            self._collect_global_stats_simple(model)
        elif self.parallel_mode.startswith('zero'):
            stage = int(self.parallel_mode[-1])  # 1,2,3
            if stage == 1:
                self._collect_global_stats_simple(model)
            else:
                self._collect_global_stats_zero2_3(model)
        elif self.parallel_mode == 'tp':
            # 简化写：假设TP的梯度已在反向里AllReduce完毕
            self._collect_global_stats_simple(model)
        elif self.parallel_mode == 'pp':
            # 流水线并行: local == global
            self._collect_global_stats_single(model)
        else:
            # 其他或混合并行
            self._collect_global_stats_simple(model)

        # ------ 写文件 ------
        if self.global_step % self.write_interval == 0:
            self._flush_to_csv_local()
            self._flush_to_csv_global()

    def _collect_global_stats_single(self, model):
        """ 对于不需要跨卡聚合的场景(local就是global)，直接写到 global 里即可。 """
        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.grad is not None:
                record = {
                    'step': self.global_step,
                    'rank': self.rank,
                    'param_name': name,
                }
                self._calc_stats(record, param.grad, suffix="")
                self.grad_records_global.append(record)

    def _collect_global_stats_simple(self, model):
        """ 对于像 DP 或 ZeRO1 / TP(已AllReduce) 等场景，param.grad 就是聚合后的梯度。 """
        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.grad is not None:
                record = {
                    'step': self.global_step,
                    'rank': self.rank,
                    'param_name': name,
                }
                self._calc_stats(record, param.grad, suffix="")
                self.grad_records_global.append(record)

    def _collect_global_stats_zero2_3(self, model):
        """
        ZeRO Stage2/3: param.grad 在某些时刻只有分片。如需全局统计，需做分布式规约。
        """
        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.grad is None:
                continue

            # local partial grads
            grad = param.grad
            # 计算本地 norm^2, max, min, sum, count ...
            local_record = {}

            if "norm" in self.stats_to_calc:
                local_record["sum_sq"] = (grad * grad).sum().item()
            if "max" in self.stats_to_calc:
                local_record["max_val"] = grad.max().item()
            if "min" in self.stats_to_calc:
                local_record["min_val"] = grad.min().item()
            if "mean" in self.stats_to_calc:
                local_record["sum_val"] = grad.sum().item()
                local_record["numel"]   = grad.numel()

            # 进行 all_reduce 收敛
            global_record = {
                'step': self.global_step,
                'rank': self.rank,
                'param_name': name,
            }

            device = grad.device
            # 1) norm
            if "norm" in self.stats_to_calc:
                sum_sq_tensor = torch.tensor([local_record["sum_sq"]], device=device)
                dist.all_reduce(sum_sq_tensor, op=dist.ReduceOp.SUM)
                global_record["norm"] = float(sum_sq_tensor.sqrt().item())

            # 2) max
            if "max" in self.stats_to_calc:
                max_val_tensor = torch.tensor([local_record["max_val"]], device=device)
                dist.all_reduce(max_val_tensor, op=dist.ReduceOp.MAX)
                global_record["max"] = float(max_val_tensor.item())

            # 3) min
            if "min" in self.stats_to_calc:
                min_val_tensor = torch.tensor([local_record["min_val"]], device=device)
                dist.all_reduce(min_val_tensor, op=dist.ReduceOp.MIN)
                global_record["min"] = float(min_val_tensor.item())

            # 4) mean
            if "mean" in self.stats_to_calc:
                sum_val_tensor = torch.tensor([local_record["sum_val"]], device=device)
                dist.all_reduce(sum_val_tensor, op=dist.ReduceOp.SUM)
                numel_tensor = torch.tensor([local_record["numel"]], device=device)
                dist.all_reduce(numel_tensor, op=dist.ReduceOp.SUM)
                if numel_tensor.item() > 0:
                    global_record["mean"] = float(sum_val_tensor.item() / numel_tensor.item())
                else:
                    global_record["mean"] = 0.0

            self.grad_records_global.append(global_record)

    def _flush_to_csv_local(self):
        """ 将本地(聚合前) grad_records_local 写入 CSV，然后清空。 """
        if not self.grad_records_local:
            return
        file_exists = os.path.exists(self.output_file_local)
        fieldnames = ['step','rank','param_name']
        for s in ["norm","mean","max","min"]:
            if s in self.stats_to_calc:
                fieldnames.append(s)
        with open(self.output_file_local, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in self.grad_records_local:
                writer.writerow(r)
        self.grad_records_local = []

    def _flush_to_csv_global(self):
        """ 将“聚合后” grad_records_global 写入 CSV，然后清空。 """
        if not self.grad_records_global:
            return
        file_exists = os.path.exists(self.output_file_global)
        fieldnames = ['step','rank','param_name']
        for s in ["norm","mean","max","min"]:
            if s in self.stats_to_calc:
                fieldnames.append(s)
        with open(self.output_file_global, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in self.grad_records_global:
                writer.writerow(r)
        self.grad_records_global = []

    def stop(self):
        """ 训练结束后，写剩余数据到文件。 """
        self._flush_to_csv_local()
        self._flush_to_csv_global()
        if self.rank == 0:
            print(f"[ColossalAIMonitor] All stats have been saved to:\n"
                  f" - local:  {self.output_file_local}\n"
                  f" - global: {self.output_file_global}")


def main():
    try_init_distributed_backend(backend='hccl')

    # Ascend NPU 检测
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 定义一个 torch.device
    if torch_npu and torch_npu.npu.is_available():
        device = torch.device(f"npu:{local_rank}")
        torch_npu.npu.set_device(device)
    else:
        device = torch.device("cpu")
        print("[WARN] Ascend NPU not available, falling back to CPU")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"[INFO] rank={rank}/{world_size}, local_rank={local_rank}, using device {device}")

    # 2) 加载配置并初始化监控
    config_path = "monitor_config.json"  
    monitor = ColossalAIMonitor(config_path=config_path)

    # 3) 构建一个简单模型 (放到 device 上)
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    # 包装为 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4) 准备模拟数据
    data = torch.rand(64, 784, device=device)
    targets = torch.randint(0, 10, (64,), device=device)

    # 5) 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 6) 注册梯度 Hook (采集本地)
    monitor.monitor(model)

    # 7) 训练循环 (这里演示单epoch)
    epochs = 1
    steps_per_epoch = 5
    for epoch in range(1, epochs + 1):
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()  # 触发Hook,记录local梯度
            ########################
            #  在 backward() 之后，测试 param.grad 是否已经全局合并
            ########################
            if dist.is_initialized() and dist.get_world_size() > 1:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    # 1) 复制一份当前的梯度
                    grad_copy = param.grad.clone()
            
                    # 2) 手动做一次全局 AllReduce(求和) + 平均
                    dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM)
                    grad_copy /= dist.get_world_size()
            
                    # 3) 对比手动 all-reduce 后的梯度 与 原来的 param.grad
                    diff = (param.grad - grad_copy).abs().max().item()
                    base = grad_copy.abs().max().item()
            
                    # 如果梯度本身特别小(基准也很小)，我们可以用绝对误差；否则可用相对误差
                    rel_err = diff / (base + 1e-8)
            
                    # 4) 打印或记录对比结果(可根据阈值判断是否一致)
                    if rel_err > 1e-5:
                        print(f"[CheckTP] Param `{name}` has large difference after manual allreduce: "
                              f"max_abs_diff={diff}, rel_err={rel_err}")
                    else:
                        print(f"[CheckTP] Param `{name}` is ALREADY (near) allreduced. diff={diff}, rel_err={rel_err}")
            optimizer.step()

            # step结束后记录 global
            monitor.step_end(model)

            if monitor.rank == 0:
                print(f"[Epoch {epoch}, Step {step+1}] loss = {loss.item():.4f}")

    # 8) 收尾
    monitor.stop()

    # 如果需要的话，也可 dist.destroy_process_group() 
    # dist.destroy_process_group()

if __name__ == '__main__':
    main()

