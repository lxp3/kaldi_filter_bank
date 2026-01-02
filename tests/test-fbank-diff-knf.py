import torch
import os
import sys

# Add the parent directory to sys.path to import kaldi_filter_bank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaldi_filter_bank.filter_bank import Filterbank

def compare_knf(onnx_compatible=False):
    """Comparison test with kaldi-native-fbank library"""
    print(f"\n[6] Comparison test with kaldi-native-fbank (onnx_compatible={onnx_compatible})")
    try:
        import kaldi_native_fbank as knf
        
        # Sampling rate and test waveform
        sample_rate = 16000
        test_waveform = torch.randn(1, 48000)  # 3 seconds of audio
        
        # Configuration
        num_mel_bins = 80
        frame_length = 25.0
        frame_shift = 10.0
        low_freq = 20.0
        high_freq = 0.0  # Nyquist
        dither = 0.0     # Disable randomness for testing
        preemph_coeff = 0.97
        
        # Our implementation
        our_fbank = Filterbank(
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            low_freq=low_freq,
            high_freq=high_freq,
            dither=dither,
            preemph_coeff=preemph_coeff,
            remove_dc_offset=True,
            snip_edges=True,
            use_energy=False,
            onnx_compatible=onnx_compatible
        )
        our_features = our_fbank(test_waveform)
        
        # knf implementation
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = sample_rate
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.dither = dither
        opts.frame_opts.preemph_coeff = preemph_coeff
        opts.frame_opts.remove_dc_offset = True
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.low_freq = low_freq
        opts.mel_opts.high_freq = high_freq if high_freq > 0 else sample_rate / 2.0
        opts.use_energy = False
        
        fbank_extractor = knf.OnlineFbank(opts)
        fbank_extractor.accept_waveform(sample_rate, test_waveform.squeeze(0).numpy())
        
        num_frames = fbank_extractor.num_frames_ready
        knf_features = []
        for i in range(num_frames):
            knf_features.append(fbank_extractor.get_frame(i))
        knf_features = torch.from_numpy(torch.stack([torch.from_numpy(f) for f in knf_features]).numpy())
        
        print(f"Our features shape: {our_features.shape}")
        print(f"knf features shape: {knf_features.shape}")
        
        # Align shapes for comparison
        our_feat = our_features.squeeze(0)
        kf_feat = knf_features
        
        min_frames = min(our_feat.shape[0], kf_feat.shape[0])
        our_feat = our_feat[:min_frames, :]
        kf_feat = kf_feat[:min_frames, :]
        
        if our_feat.shape != kf_feat.shape:
            print(f"✗ Shape mismatch! {our_feat.shape} vs {kf_feat.shape}")
            return False
            
        max_diff = torch.abs(our_feat - kf_feat).max().item()
        mean_diff = torch.abs(our_feat - kf_feat).mean().item()
        
        print(f"All frames - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        # Tolerance check
        tolerance = 1e-3
        if max_diff < tolerance:
            print(f"✓ kaldi-native-fbank comparison passed! (Max diff < {tolerance})")
            return True
        else:
            print(f"✗ kaldi-native-fbank comparison failed! Difference exceeds tolerance {tolerance}")
            return False
            
    except ImportError:
        print("kaldi-native-fbank not installed, skipping comparison test")
        return True
    except Exception as e:
        print(f"kaldi-native-fbank comparison error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    compare_knf(onnx_compatible=False)
    compare_knf(onnx_compatible=True)
