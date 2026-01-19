import torch
import torch.nn as nn
import os

class STFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 预先计算 window 并注册为 buffer，确保导出到 ONNX 时 window 作为常量或输入存在
        self.register_buffer('window', torch.hann_window(512).pow(0.5))

    def forward(self, x):
        # 匹配用户提供的代码行
        # x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
        return torch.stft(x, 512, 256, 512, self.window, return_complex=False)

class ISTFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(512).pow(0.5))

    def forward(self, y):
        # 匹配用户提供的代码行
        # y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5))
        # 注意：istft 通常期望输入 y 为复数张量。
        # 如果 y 是 stft(return_complex=False) 的输出，形状为 [..., 2]，需要先转为 complex
        if not torch.is_complex(y) and y.shape[-1] == 2:
            y = torch.view_as_complex(y)
        return torch.istft(y, 512, 256, 512, self.window)

def test_export():
    print(f"Torch version: {torch.__version__}")
    
    # 1. 测试 STFT 导出
    print("\n--- Testing STFT ONNX export ---")
    stft_model = STFTModel()
    stft_model.eval()
    x_dummy = torch.randn(1, 16000)
    try:
        torch.onnx.export(
            stft_model, x_dummy, "stft.onnx",
            opset_version=17,
            input_names=['input'], output_names=['output']
        )
        print("STFT ONNX export SUCCESSFUL")
    except Exception as e:
        print(f"STFT ONNX export FAILED: {e}")

    # 2. 测试 ISTFT 导出
    print("\n--- Testing ISTFT ONNX export ---")
    istft_model = ISTFTModel()
    istft_model.eval()
    # freq = 512 // 2 + 1 = 257, time frames depends on signal length
    y_dummy = torch.randn(1, 257, 63, 2) # 模拟 return_complex=False 的输出
    try:
        torch.onnx.export(
            istft_model, y_dummy, "istft.onnx",
            opset_version=17,
            input_names=['input'], output_names=['output']
        )
        print("ISTFT ONNX export SUCCESSFUL")
    except Exception as e:
        print(f"ISTFT ONNX export FAILED: {e}")

    # 清理生成的临时文件
    for f in ["stft.onnx", "istft.onnx"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_export()
