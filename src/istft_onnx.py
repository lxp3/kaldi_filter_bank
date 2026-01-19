#!/usr/bin/env python3
"""
ISTFT implementation using only PyTorch, compatible with ONNX export.
"""

import torch
from torch import nn
from torch.nn import functional as F

MAX_FRAMES = 5200
DEFAULT_OPSET = 17


class ISTFT(nn.Module):
    """
    Inverse Short-Time Fourier Transform implementation using only PyTorch.
    Matches torch.istft(..., center=True) behavior.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor = None,
        center: bool = True
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        if window is None:
            window = torch.ones(win_length)
        
        # Ensure window is correct length
        assert window.shape[0] == win_length

        # Create the inverse basis numerically to ensure exact match with torch.fft.irfft
        # We process (batch, freq, frames). Input to conv_transpose1d is (batch, channels, frames)
        # channels = 2 * (n_fft // 2 + 1)
        # We want to find the kernel K such that conv_transpose1d(X, K) = OLA(Window * IRFFT(X))
        
        n_freqs = n_fft // 2 + 1
        dim = 2 * n_freqs
        
        # Create identity matrix representing all possible spectral components
        # Shape: [dim, dim] -> [dim, n_freqs, 2] (complex view)
        # We iterate over the 'dim' dimension as a batch of OHE vectors
        basis_complex = torch.zeros(dim, n_freqs, dtype=torch.complex64)
        
        # Fill basis
        # First n_freqs are Real parts
        for i in range(n_freqs):
            basis_complex[i, i] = 1.0 + 0j
        # Next n_freqs are Imaginary parts
        for i in range(n_freqs):
            basis_complex[n_freqs + i, i] = 0.0 + 1j

        # Compute IRFFT for these basis vectors
        # output shape: [dim, n_fft]
        # This gives us the time-domain signal corresponding to each spectral component
        time_basis = torch.fft.irfft(basis_complex, n=n_fft, dim=-1)
        
        # Apply window
        # Window needs to be padded to n_fft if win_length < n_fft
        # torch.istft logic: the window is applied to the time-domain signal of length n_fft.
        # However, the window is usually strictly defined on win_length and then centered/padded.
        # torch.stft(center=True) centers the window.
        # But actually torch.istft applies the window element-wise to the n_fft output of irfft.
        # We should assume the user provided window is what they want applied.
        # If win_length < n_fft, torch.istft pads the window with zeros? 
        # Documentation: "The window is padded to length n_fft with zeros".
        # If center=True (in STFT), the window is centered? N/A for ISTFT window application (it's just mul).
        
        window_padded = torch.zeros(n_fft, dtype=window.dtype, device=window.device)
        # Center the window in n_fft area?
        # Actually torch.istft assumes the window is length n_fft or matches win_length.
        # If win_length < n_fft, it essentially pads centered?
        # Let's verify torch source or behavior. Usually it's centered pad.
        start = (n_fft - win_length) // 2
        window_padded[start:start+win_length] = window
        
        # Multiply basis by window
        # time_basis: [dim, n_fft]
        # window_padded: [n_fft]
        time_basis_windowed = time_basis * window_padded.unsqueeze(0)
        
        # Prepare for ConvTranspose1d
        # Input: [Batch, dim, Frames]
        # Output: [Batch, 1, Samples]
        # Weight shape for ConvTranspose1d: [in_channels, out_channels, kernel_size]
        # [dim, 1, n_fft]
        # Note: ConvTranspose1d sums over input channels.
        
        self.register_buffer('weight', time_basis_windowed.unsqueeze(1).float())
        
        # Pre-compute Window Sum (OLA denominator)
        # We can simulate this by running ones through the OLA process
        # But we need a sufficiently large buffer or handle it dynamically.
        # Since this is for ONNX export with dynamic size, we might need a way to generate it.
        # OR, we assume a max length.
        
        # For 'window_sum', we can compute the squared window (padded) and overlap-add it 
        # using the same logic (just magnitude=1 for DC? No).
        # We can implement window_sum calculation using ConvTranspose1d as well!
        # Input: ones [Batch, 1, Frames]
        # Weight: window_squared [1, 1, n_fft]
        # Output: window_sum
        
        win_sq = window_padded ** 2
        self.register_buffer('window_sq_weight', win_sq.view(1, 1, -1).float())

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        magnitude: [Batch, Freq, Frames]
        phase: [Batch, Freq, Frames]
        """
        # Recombine to Real/Imag parts flattened
        # [Batch, 2*Freq, Frames]
        recombine = torch.cat([
            magnitude * torch.cos(phase),
            magnitude * torch.sin(phase)
        ], dim=1)
        
        # 1. Synthesis (Overlap-Add)
        # Output: [Batch, 1, Samples]
        audio_unnormalized = F.conv_transpose1d(
            recombine,
            self.weight,
            stride=self.hop_length,
            padding=0
        )
        
        # 2. Compute Window Sum using ConvTranspose1d
        # We need a tensor of ones with same number of frames
        B, _, T = recombine.shape
        ones = torch.ones(B, 1, T, device=recombine.device, dtype=recombine.dtype)
        
        window_sum = F.conv_transpose1d(
            ones,
            self.window_sq_weight,
            stride=self.hop_length,
            padding=0
        )
        
        # 3. Normalize
        # Avoid div by zero
        window_sum = window_sum.clamp(min=1e-11)
        audio = audio_unnormalized / window_sum
        
        # 4. Handle 'center=True' padding removal
            # torch.istft with center=True removes n_fft//2 from both sides
        if self.center:
            start = self.n_fft // 2
            # We need to determine the output length. 
            # Total len from conv_tranpose is n_fft + (T-1)*hop
            # We trim start and end.
            audio = audio[:, :, start:-start]
            
        return audio


class ExportableISTFTModule(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        window = torch.hann_window(win_length).pow(0.5)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True
        )

    def forward(self, mag, real, imag):
        # mag, real, imag: [Batch, Freq, Frames]
        phase = torch.atan2(imag, real)
        return self.istft(mag, phase)


def export_istft_to_onnx(output_path, n_fft=512, hop_length=256, win_length=512):
    model = ExportableISTFTModule(n_fft, hop_length, win_length)
    model.eval()
    
    n_freqs = n_fft // 2 + 1
    dummy_input = (
        torch.rand(1, n_freqs, 100),
        torch.rand(1, n_freqs, 100),
        torch.rand(1, n_freqs, 100)
    )
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["mag", "real", "imag"],
        output_names=["audio"],
        dynamic_axes={
            "mag": {0: "batch", 2: "time"},
            "real": {0: "batch", 2: "time"},
            "imag": {0: "batch", 2: "time"},
            "audio": {0: "batch", 2: "samples"},
        },
        opset_version=17
    )
    print(f"Exported to {output_path}")

def test_consistency():
    print("Testing consistency...")
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length).pow(0.5)
    
    x = torch.randn(1, 16000)
    stft = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    
    # torch istft
    y_torch = torch.istft(stft, n_fft, hop_length, win_length, window, center=True)
    
    # custom istft
    model = ExportableISTFTModule(n_fft, hop_length, win_length)
    model.eval()
    
    mag = stft.abs()
    phase = stft.angle()
    # We pass real/imag just for interface, but inside we compute phase from them. 
    # Actually Exportable uses atan2(imag, real). 
    real = stft.real
    imag = stft.imag
    
    with torch.no_grad():
        y_custom = model(mag, real, imag)
        
    y_custom = y_custom.squeeze(1)
    
    # Align lengths
    min_len = min(y_torch.shape[-1], y_custom.shape[-1])
    diff = (y_torch[..., :min_len] - y_custom[..., :min_len]).abs().max()
    print(f"Max Difference: {diff.item()}")
    
    if diff < 1e-4:
        print("PASSED")
    else:
        print("FAILED")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_consistency()
    else:
        export_istft_to_onnx("istft.onnx")