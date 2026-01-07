import torch
import os
import sys

# Add the parent directory to sys.path to import kaldi_filter_bank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaldi_filter_bank.filter_bank import Filterbank

def test_export_onnx():
    """Test ONNX export functionality"""
    print("\n[3] ONNX Export Test")
    try:
        fbank = Filterbank(
            sample_rate=16000,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            low_freq=20.0,
            high_freq=8000.0,
            dither=0.0,  # Disable dither for export
            preemph_coeff=0.97,
            remove_dc_offset=True,
            snip_edges=True,
            use_energy=False,
        )
        fbank.eval()
        dummy_input = torch.randn(1, 16000)
        
        output_path = "filterbank.onnx"
        torch.onnx.export(
            fbank,
            dummy_input,
            output_path,
            input_names=['waveform'],
            output_names=['fbank_features'],
            dynamic_axes={
                'waveform': {0: 'batch', 1: 'time'},
                'fbank_features': {0: 'batch', 1: 'frames'}
            },
            opset_version=17,
        )
        print(f"✓ ONNX export successful! File: {output_path}")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    test_export_onnx()
