"""
Test script to verify ONNX Runtime can correctly run STFT and ISTFT models.
This script:
1. Exports PyTorch STFT/ISTFT models to ONNX
2. Loads them with ONNX Runtime
3. Verifies inference results match PyTorch output
"""
import torch
import torch.nn as nn
import numpy as np
import os

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    exit(1)

class STFTModel(nn.Module):
    """STFT model wrapper for ONNX export."""
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length).pow(0.5))

    def forward(self, x):
        return torch.stft(
            x, self.n_fft, self.hop_length, self.win_length, 
            self.window, return_complex=False
        )

class ISTFTModel(nn.Module):
    """ISTFT model wrapper for ONNX export."""
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length).pow(0.5))

    def forward(self, y):
        # Convert real tensor [B, F, T, 2] to complex [B, F, T]
        if not torch.is_complex(y) and y.shape[-1] == 2:
            y = torch.view_as_complex(y)
        return torch.istft(
            y, self.n_fft, self.hop_length, self.win_length, self.window
        )

def export_model(model, dummy_input, onnx_path, opset=17):
    """Export a PyTorch model to ONNX format."""
    model.eval()
    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    return os.path.exists(onnx_path)

def run_ort_inference(onnx_path, input_np):
    """Run ONNX Runtime inference."""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'input': input_np})
    return outputs[0]

def test_stft_inference():
    """Test STFT ONNX export and ORT inference."""
    print("\n" + "="*50)
    print("STFT ONNX Runtime Inference Test")
    print("="*50)
    
    onnx_path = "test_stft.onnx"
    model = STFTModel()
    model.eval()
    
    # Create test input: 1 second of audio at 16kHz
    x = torch.randn(1, 16000)
    
    # Get PyTorch reference output
    with torch.no_grad():
        torch_out = model(x)
    
    # Export to ONNX
    print("Exporting STFT to ONNX...")
    try:
        export_model(model, x, onnx_path)
        print(f"  Export successful: {onnx_path}")
    except Exception as e:
        print(f"  Export FAILED: {e}")
        return False
    
    # Run ORT inference
    print("Running ONNX Runtime inference...")
    try:
        ort_out = run_ort_inference(onnx_path, x.numpy())
        print(f"  Inference completed")
        print(f"  Output shape: {ort_out.shape}")
        
        # Compare results
        max_diff = np.abs(ort_out - torch_out.numpy()).max()
        mean_diff = np.abs(ort_out - torch_out.numpy()).mean()
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        if max_diff < 1e-4:
            print("  Result: PASSED")
            return True
        else:
            print("  Result: MISMATCH (difference too large)")
            return False
    except Exception as e:
        print(f"  Inference FAILED: {e}")
        return False
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_istft_inference():
    """Test ISTFT ONNX export and ORT inference."""
    print("\n" + "="*50)
    print("ISTFT ONNX Runtime Inference Test")
    print("="*50)
    
    onnx_path = "test_istft.onnx"
    stft_model = STFTModel()
    istft_model = ISTFTModel()
    stft_model.eval()
    istft_model.eval()
    
    # Create STFT output as ISTFT input
    x = torch.randn(1, 16000)
    with torch.no_grad():
        stft_out = stft_model(x)  # shape: [1, 257, T, 2]
        torch_out = istft_model(stft_out)
    
    # Export to ONNX
    print("Exporting ISTFT to ONNX...")
    try:
        export_model(istft_model, stft_out, onnx_path)
        print(f"  Export successful: {onnx_path}")
    except Exception as e:
        print(f"  Export FAILED: {e}")
        return False
    
    # Run ORT inference
    print("Running ONNX Runtime inference...")
    try:
        ort_out = run_ort_inference(onnx_path, stft_out.numpy())
        print(f"  Inference completed")
        print(f"  Output shape: {ort_out.shape}")
        
        # Compare results
        max_diff = np.abs(ort_out - torch_out.numpy()).max()
        mean_diff = np.abs(ort_out - torch_out.numpy()).mean()
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        if max_diff < 1e-4:
            print("  Result: PASSED")
            return True
        else:
            print("  Result: MISMATCH (difference too large)")
            return False
    except Exception as e:
        print(f"  Inference FAILED: {e}")
        return False
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX Runtime version: {ort.__version__}")
    
    stft_ok = test_stft_inference()
    istft_ok = test_istft_inference()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"  STFT:  {'PASSED' if stft_ok else 'FAILED'}")
    print(f"  ISTFT: {'PASSED' if istft_ok else 'FAILED'}")
    
    if stft_ok and istft_ok:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    main()
