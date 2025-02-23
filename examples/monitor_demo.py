import torch
import torch.nn as nn
import torch.optim as optim
import torch_npu

class ColossalAIMonitor:
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        # 用于存放每次梯度统计的记录
        self.grad_records = []

    def _create_hook_fn(self, param_name):
        def hook_fn(grad):
            # 计算常见的四种统计信息
            norm_val = grad.norm().item()
            max_val = grad.max().item()
            min_val = grad.min().item()
            mean_val = grad.mean().item()

            self.grad_records.append({
                'param_name': param_name,
                'norm': norm_val,
                'max': max_val,
                'min': min_val,
                'mean': mean_val
            })
        return hook_fn

    def monitor(self, model):
        """
        用户在训练前调用此方法，为 model 中的所有需要监控的参数注册梯度 Hook。
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._create_hook_fn(name))

    def stop(self):
        import csv
        output_file = self.config.get('output_file', 'grad_stats.csv')
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['param_name', 'norm', 'max', 'min', 'mean'])
            writer.writeheader()
            for record in self.grad_records:
                writer.writerow(record)

        print(f"Gradient stats have been saved to {output_file}")

def main():
    # -----------------------------------------------------------------------
    # 1. 选择设备：若 torch_npu.npu.is_available() 返回 True 则使用 'npu:0'，否则用 CPU
    # -----------------------------------------------------------------------
    if torch_npu.npu.is_available():
        device = torch.device("npu:0")
        print("NPU is available. Using device:", device)
    else:
        device = torch.device("cpu")
        print("NPU not available. Falling back to CPU.")

    # -----------------------------------------------------------------------
    # 2. 创建一个简单的模型（全连接网络），并把它搬到指定的 device
    # -----------------------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    # -----------------------------------------------------------------------
    # 3. 定义简单的数据和目标，演示一次前向/反向训练
    # -----------------------------------------------------------------------
    data = torch.rand(64, 784).to(device)   # 64条数据，每条784维输入
    targets = torch.randint(0, 10, (64,)).to(device)  # 随机 10 分类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # -----------------------------------------------------------------------
    # 4. 创建并启用监控
    # -----------------------------------------------------------------------
    monitor = ColossalAIMonitor()
    monitor.monitor(model)

    # -----------------------------------------------------------------------
    # 5. 简单的训练循环，演示 Autograd Hook 的触发
    # -----------------------------------------------------------------------
    epochs = 2
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()   # <--- 在此触发 Hook，收集梯度统计
        optimizer.step()
        print(f"[Epoch {epoch}/{epochs}] loss = {loss.item():.4f}")

    # -----------------------------------------------------------------------
    # 6. 训练结束后收集并输出统计数据
    # -----------------------------------------------------------------------
    monitor.stop()

if __name__ == '__main__':
    main()
