#!/usr/bin/env python

import os
import sys
import torch


# Add the parent directory to sys.path to import kaldi_filter_bank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaldi_filter_bank.filter_bank import Filterbank

def compare_kaldifeat():
    """Comparison test with kaldifeat library"""
    print(f"\n[4] Comparison test with kaldifeat")
    try:
        import kaldifeat
        
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
        )
        our_features = our_fbank(test_waveform)
        
        # kaldifeat implementation
        opts = kaldifeat.FbankOptions()
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
        
        fbank_extractor = kaldifeat.Fbank(opts)
        
        # kaldifeat expects [num_samples] input
        kaldifeat_features = fbank_extractor(test_waveform.squeeze(0))
        
        print(f"Our features shape: {our_features.shape}")
        print(f"kaldifeat features shape: {kaldifeat_features.shape}")
        
        # Align shapes for comparison
        our_feat = our_features.squeeze(0)
        kf_feat = kaldifeat_features
        
        min_frames = min(our_feat.shape[0], kf_feat.shape[0])
        our_feat = our_feat[:min_frames, :]
        kf_feat = kf_feat[:min_frames, :]
        
        if our_feat.shape != kf_feat.shape:
            print(f"✗ Shape mismatch! {our_feat.shape} vs {kf_feat.shape}")
            return False
            
        max_diff = torch.abs(our_feat - kf_feat).max().item()
        mean_diff = torch.abs(our_feat - kf_feat).mean().item()
        
        # Skip boundary frames if necessary
        if min_frames > 2:
            our_feat_mid = our_feat[1:-1, :]
            kf_feat_mid = kf_feat[1:-1, :]
            max_diff_mid = torch.abs(our_feat_mid - kf_feat_mid).max().item()
            mean_diff_mid = torch.abs(our_feat_mid - kf_feat_mid).mean().item()
        else:
            max_diff_mid = max_diff
            mean_diff_mid = mean_diff
        
        print(f"All frames - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"Middle frames - Max diff: {max_diff_mid:.6f}, Mean diff: {mean_diff_mid:.6f}")
        
        # Tolerance check
        tolerance = 1e-3
        if max_diff_mid < tolerance:
            print(f"✓ kaldifeat comparison passed! (Max diff < {tolerance})")
            return True
        else:
            print(f"✗ kaldifeat comparison failed! Difference exceeds tolerance {tolerance}")
            return False
            
    except ImportError:
        print("kaldifeat not installed, skipping comparison test")
        return True
    except Exception as e:
        print(f"kaldifeat comparison error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    compare_kaldifeat()
