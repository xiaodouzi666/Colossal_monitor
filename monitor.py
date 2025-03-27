import os
import json
import csv
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import colossalai
    from colossalai.nn.parallel import ZeroDDP
    COLOSSALAI_AVAILABLE = True
except ImportError:
    COLOSSALAI_AVAILABLE = False
    ZeroDDP = None  # 占位，避免NameError


def distributed_is_initialized():
    """ 检查是否已经在分布式环境中 (PyTorch层面) """
    return dist.is_available() and dist.is_initialized()


class ColossalAIMonitor:
    def __init__(self, config_path=None):
        """
        :param config_path: JSON 文件路径，可包含如下字段：
            - output_file_local / output_file_global
            - stats: ["norm", "mean", "max", "min"] 等
            - parallel_mode: "dp" / "zero2" / "zero3" / "tp" / "pp" / "none" / etc.
            - write_interval: 每隔多少 step 写一次 CSV
            - param_list: 要监控的参数名列表（可选）
        """
        self.config = self._load_config(config_path)

        # 记录本地和全局梯度（聚合前 / 聚合后）
        self.grad_records_local = []
        self.grad_records_global = []

        # 保存步数
        self.global_step = 0

        # 从配置文件/字典获取各种设置
        self.output_file_local  = self.config.get('output_file_local',  'grad_stats_local.csv')
        self.output_file_global = self.config.get('output_file_global', 'grad_stats_global.csv')
        self.param_list = self.config.get('param_list', [])   # 如果不为空则只监控指定参数
        self.stats_to_calc = self.config.get('stats', ["norm", "mean", "max", "min"])
        self.write_interval = self.config.get('write_interval', 1)

        # 注意：parallel_mode 可以在外部写在 config 中，
        # 也可以后续根据模型类型(ZeroDDP)来自动检测再赋值
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

    def monitor(self, model):
        """
        用户在训练脚本中显式调用，用于：
          1. 为 model 中的可学习参数注册 Hook (采集本地梯度)
          2. 如果需要，也可在这里检测 ColossalAI 并行类型
        """
        # 如果没指定 parallel_mode，或者是 'auto'，可以做自动检测
        # 这里是示范：若使用 ZeroDDP 则设 parallel_mode='zero'
        # 你可根据实际需要加强判断 (例如 zero2 / zero3 / tp / pp)
        detected_mode = self._detect_parallel_mode_from_model(model)
        if detected_mode and self.parallel_mode == 'dp':
            # 仅当用户未自定义 parallel_mode='dp' 时，才覆盖
            self.parallel_mode = detected_mode

        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.requires_grad:
                param.register_hook(self._create_hook_fn(name))

        print(f"[ColossalAIMonitor] Hook registration done. parallel_mode={self.parallel_mode}")

    def _detect_parallel_mode_from_model(self, model):
        """
        如果想自动识别 ColossalAI 并行模式，可以在这里加更复杂的逻辑。
        现仅示范：若 model 是 ZeroDDP 就返回 'zero2/3'，否则返回 None。
        你也可检测 model.zero_stage, model.tp_degree 等信息...
        """
        if COLOSSALAI_AVAILABLE and isinstance(model, ZeroDDP):
            # demo: 假设 zero_stage=2 or 3
            stage = getattr(model, 'zero_stage', None)
            if stage == 2:
                return 'zero2'
            elif stage == 3:
                return 'zero3'
            else:
                return 'zero'
        # 还可判断 tensor parallel, pipeline parallel ...
        return None

    def _create_hook_fn(self, param_name):
        """在“本地梯度”计算完时，记录一次统计(聚合前)."""
        def hook_fn(grad):
            # grad: 这里是 "聚合前" 的梯度(若DDP的话, AllReduce还没发生; ZeRO/TP可能已经过通信).
            # 你可以在此处 grad.detach() 或 .cpu() 避免占用显存过多
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
        # 注意：在大模型场景下，.item() / .max() 这些会触发同步，可考虑只抽样或只每N步统计
        if "norm" in self.stats_to_calc:
            record_dict["norm"+suffix] = float(grad_tensor.norm().detach().cpu())
        if "mean" in self.stats_to_calc:
            record_dict["mean"+suffix] = float(grad_tensor.mean().detach().cpu())
        if "max" in self.stats_to_calc:
            record_dict["max"+suffix] = float(grad_tensor.max().detach().cpu())
        if "min" in self.stats_to_calc:
            record_dict["min"+suffix] = float(grad_tensor.min().detach().cpu())

    def step_end(self, model):
        """
        每步训练完成后由用户显式调用。
        - 先增加 step 计数；
        - 再根据并行模式，对“聚合后”的梯度进行采集；
        - 若到达写文件间隔，就写到CSV。
        """
        self.global_step += 1

        # 采集 "全局" 梯度
        if self.parallel_mode in ('none', 'pp'):
            # 这两个模式下 local == global
            self._collect_global_stats_single(model)
        elif self.parallel_mode == 'dp':
            self._collect_global_stats_simple(model)
        elif self.parallel_mode.startswith('zero'):
            stage_num = 2 if '2' in self.parallel_mode else 3
            self._collect_global_stats_zero2_3(model, stage=stage_num)
        elif self.parallel_mode == 'tp':
            self._collect_global_stats_simple(model)
        else:
            # 其他或者是混合并行
            self._collect_global_stats_simple(model)

        # 到达 write_interval 步时写出文件
        if self.global_step % self.write_interval == 0:
            self._flush_to_csv_local()
            self._flush_to_csv_global()

    def _collect_global_stats_single(self, model):
        """
        不需要跨卡聚合的场景 (single GPU / pipeline parallel):
        直接取 param.grad 即可。
        """
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
        """
        在纯DP或TP里, PyTorch DDP(默认)会在 backward 完成后做AllReduce，
        因此 param.grad 已经是全局梯度.
        对TP某些情况(列/行并行)可能只是本片Grad, 这里示例仍然做一次 gather/或统计.
        """
        # 注意: 如果是张量并行+ColossalAI,
        #       param.grad 可能只包含一部分, 需自定义all_reduce或shard gather
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

    def _collect_global_stats_zero2_3(self, model, stage=2):
        """
        ZeRO Stage2/3 下, param.grad 可能是分片形态, 
        或者已经局部/全局聚合过, 取决于 ColossalAI ZeroDDP 的实现.
        这里保留原先你写的手动 all_reduce 逻辑做统计.

        若要真正兼容 ColossalAI Zero3, 
        可以检查 param 是否在本地，或者param.colo_attr.sharded_data_tensor 是否存在,
        并调用 chunk_manager 等获取正确的分片梯度. 
        (示例中仅演示, 未做对 chunk gather.)
        """
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            if self.param_list and name not in self.param_list:
                continue
            if param.grad is None:
                continue

            grad = param.grad
            local_record = {}

            if "norm" in self.stats_to_calc:
                sum_sq = (grad * grad).sum().detach()
                local_record["sum_sq"] = sum_sq

            if "max" in self.stats_to_calc:
                local_record["max_val"] = grad.max().detach()

            if "min" in self.stats_to_calc:
                local_record["min_val"] = grad.min().detach()

            if "mean" in self.stats_to_calc:
                local_record["sum_val"] = grad.sum().detach()
                local_record["numel"]   = grad.numel()

            # all_reduce 方式收集全局统计 (注意: 真实ZeRO3可分 shard)
            global_record = {
                'step': self.global_step,
                'rank': self.rank,
                'param_name': name,
            }

            # 1) norm
            if "norm" in self.stats_to_calc:
                sum_sq_tensor = torch.tensor([local_record["sum_sq"].item()], device=device)
                dist.all_reduce(sum_sq_tensor, op=dist.ReduceOp.SUM)
                global_record["norm"] = float(sum_sq_tensor.sqrt().item())

            # 2) max
            if "max" in self.stats_to_calc:
                max_val_tensor = torch.tensor([local_record["max_val"].item()], device=device)
                dist.all_reduce(max_val_tensor, op=dist.ReduceOp.MAX)
                global_record["max"] = float(max_val_tensor.item())

            # 3) min
            if "min" in self.stats_to_calc:
                min_val_tensor = torch.tensor([local_record["min_val"].item()], device=device)
                dist.all_reduce(min_val_tensor, op=dist.ReduceOp.MIN)
                global_record["min"] = float(min_val_tensor.item())

            # 4) mean
            if "mean" in self.stats_to_calc:
                sum_val_tensor = torch.tensor([local_record["sum_val"].item()], device=device)
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
        """
        训练结束后，写剩余数据到文件，并打印提示。
        """
        self._flush_to_csv_local()
        self._flush_to_csv_global()
        if self.rank == 0:
            print(f"[ColossalAIMonitor] All stats have been saved to:\n"
                  f" - local:  {self.output_file_local}\n"
                  f" - global: {self.output_file_global}")
