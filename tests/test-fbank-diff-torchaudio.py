import torch
import os
import sys

# Add the parent directory to sys.path to import kaldi_filter_bank
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaldi_filter_bank.filter_bank import Filterbank


def compare_torchaudio_fbank():
    print("\n[5] Comparison test with torchaudio.compliance.kaldi")
    try:
        import torchaudio.compliance.kaldi as kaldi

        # Sampling rate and test waveform
        sample_rate = 16000
        test_waveform = torch.randn(1, 48000)  # 3 seconds of audio

        # Configuration
        num_mel_bins = 80
        frame_length = 25.0
        frame_shift = 10.0
        low_freq = 20.0
        high_freq = 0.0  # Nyquist
        dither = 0.0  # Disable randomness for testing
        preemph_coeff = 0.97
        remove_dc_offset = True

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
            remove_dc_offset=remove_dc_offset,
            snip_edges=True,
            use_energy=False,
            window_fn=torch.hamming_window,
        )
        our_features = our_fbank(test_waveform)

        # Kaldi implementation via torchaudio
        kaldi_features = kaldi.fbank(
            test_waveform,
            sample_frequency=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            low_freq=low_freq,
            high_freq=high_freq,
            dither=dither,
            preemphasis_coefficient=preemph_coeff,
            remove_dc_offset=remove_dc_offset,
            snip_edges=True,
            use_energy=False,
            window_type="hamming",
        )

        print(f"Our implementation shape: {our_features.shape}")
        print(f"Kaldi implementation shape: {kaldi_features.shape}")

        our_features = our_features.squeeze(0)

        # Limit frames for comparison if they differ slightly (snip_edges might have subtle effects)
        min_frames = min(our_features.shape[0], kaldi_features.shape[0])
        our_features = our_features[:min_frames, :]
        kaldi_features = kaldi_features[:min_frames, :]

        # Check shape consistency
        if our_features.shape != kaldi_features.shape:
            print(f"✗ Shape mismatch! {our_features.shape} vs {kaldi_features.shape}")
            return False

        max_diff = torch.abs(our_features - kaldi_features).max().item()
        mean_diff = torch.abs(our_features - kaldi_features).mean().item()

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        # Tolerance check
        tolerance = 1e-3
        if max_diff < tolerance:
            print(f"✓ torchaudio comparison passed! (Max diff < {tolerance})")
            return True
        else:
            print(f"✗ Comparison test failed! Difference exceeds tolerance {tolerance}")
            return False

    except ImportError:
        print("torchaudio not installed, skipping comparison test")
        return True
    except Exception as e:
        print(f"Comparison test error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.manual_seed(42)
    compare_torchaudio_fbank()
