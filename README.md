# Torch Filter Bank

An **ONNX-exportable** Mel Filterbank implementation in PyTorch, meticulously aligned with Kaldi's feature extraction behavior.

This project was developed with the assistance of **Claude 3.5 Opus** and **Antigravity**.

## Core Features

- **Kaldi Consistency**: Verified against `kaldifeat` and `kaldi-native-fbank` for numerical accuracy.
- **ONNX Ready**: Implements an ONNX-compatible signal processing chain (DFT-based FFT and `gather`-based framing) for seamless model export and deployment.
- **Pure PyTorch**: No custom C++ extensions required, ensuring high portability across platforms.

## Usage

### Simple Call
For standard feature extraction in PyTorch:

```python
import torch
from kaldi_filter_bank.filter_bank import Filterbank

fbank = Filterbank()
waveform = torch.randn(1, 16000)  # [batch, samples]
features = fbank(waveform)        # [batch, frames, mel_bins]
```

### Integrating for ONNX Export
To include the filterbank as a front-end layer in your ASR model for ONNX export:

```python
from kaldi_filter_bank.filter_bank import Filterbank

class AsrModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # Use onnx_compatible=True for DFT-based FFT and dynamic framing
        self.fbank = Filterbank(onnx_compatible=True)
        self.model = model

    def forward(self, waveforms):
        # waveforms shape: [batch, time]
        features = self.fbank(waveforms)
        logits = self.model(features)
        return logits
```
> See `tests/test-fbank-onnx-export.py` for a full export example.

## Directory Structure

- `kaldi_filter_bank/`: Core implementation of the `Filterbank` module.
- `tests/`: Comprehensive test suite for validation and comparison.
    - `test-fbank-diff-kaldifeat.py`: Numerical alignment with `kaldifeat`.
    - `test-fbank-diff-knf.py`: Numerical alignment with `kaldi-native-fbank`.
    - `test-fbank-diff-torchaudio.py`: Comparison with `torchaudio`.
    - `test-fbank-onnx-export.py`: ONNX export and verification script.

## Acknowledgments

This implementation draws inspiration and verification from:
- [kaldifeat](https://github.com/csukuangfj/kaldifeat)
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)