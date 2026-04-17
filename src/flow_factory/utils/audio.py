# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/utils/audio.py
"""
Audio utility functions for converting between waveform Tensors and NumPy arrays.

Audio in this module is always represented as **waveform** data (time-domain samples),
not spectrograms or mel features. For spectrogram-based processing, use the model's
VAE or vocoder directly.

Type Hierarchy:
    AudioSingle                             Single audio clip
        ├─ torch.Tensor (C, T)              Waveform tensor: C channels, T samples
        ├─ torch.Tensor (T,)                Mono waveform tensor (auto-promoted to (1, T))
        └─ np.ndarray (C, T) or (T,)        Waveform array

    AudioBatch                              Batch of audio clips
        ├─ torch.Tensor (B, C, T)           Uniform batch: same length
        ├─ np.ndarray (B, C, T)             Uniform batch: same length
        ├─ List[torch.Tensor]               Ragged batch: variable C/T per clip
        └─ List[np.ndarray]                 Ragged batch: variable C/T per clip

Tensor/Array Conventions:
    - torch.Tensor: Channel-first (C, T) or (B, C, T)
    - np.ndarray: Channel-first (C, T) or (B, C, T)
    - Channels: C ∈ {1, 2} for mono, stereo
    - Unlike images/video, both torch and numpy use the **same** axis order

Value Ranges:
    - [-1.0, 1.0]: Standard normalized float format (PyTorch / diffusers convention)
    - [-32768, 32767]: Standard int16 format (WAV file convention)

Main Functions:
    Type Validation:
        - is_audio(), is_audio_batch()

    Loading / Saving:
        - load_audio(): Load audio file to waveform tensor
        - save_audio(): Save waveform tensor to audio file

    Conversions:
        - audio_to_tensor(), audio_to_numpy()
        - convert_audio(): Resample and/or change channel count

    Standardization:
        - standardize_audio_batch(): Unified conversion to np/pt formats

    Hashing:
        - hash_audio(), hash_audio_list()

Examples:
    >>> # Load and standardize
    >>> waveform = load_audio("speech.wav", sample_rate=16000)
    >>> waveform.shape
    torch.Size([1, 16000])  # 1 second of mono audio at 16kHz

    >>> # Batch standardization
    >>> clips = [torch.randn(1, 16000), torch.randn(2, 32000)]
    >>> batch = standardize_audio_batch(clips, output_type='pt')
    >>> len(batch)  # Returns list for variable shapes
    2
"""

import hashlib
from pathlib import Path
from typing import List, Union, Any, Literal, Optional, Tuple

import torch
import numpy as np


# ----------------------------------- Type Aliases --------------------------------------

AudioSingle = Union[torch.Tensor, np.ndarray]
"""Type alias for a single audio waveform. Tensor shape (C, T) or (T,); array shape (C, T) or (T,)."""

AudioBatch = Union[
    torch.Tensor,           # (B, C, T)
    np.ndarray,             # (B, C, T)
    List[torch.Tensor],     # List of (C, T) — variable length allowed
    List[np.ndarray],       # List of (C, T) — variable length allowed
]
"""Type alias for a batch of audio waveforms."""


__all__ = [
    # Type aliases
    'AudioSingle',
    'AudioBatch',
    # Validation
    'is_audio',
    'is_audio_batch',
    # Loading / Saving
    'load_audio',
    'save_audio',
    # Conversions
    'audio_to_tensor',
    'audio_to_numpy',
    'convert_audio',
    # Standardization
    'standardize_audio_batch',
    # Hashing
    'hash_audio',
    'hash_audio_list',
]


# ----------------------------------- Validation --------------------------------------

def is_audio(audio: Any) -> bool:
    """
    Check if the input is a valid single audio waveform.
    Corresponds to type ``AudioSingle``.

    Args:
        audio: Input to check.

    Returns:
        bool: True if valid audio:
            - torch.Tensor: Shape (C, T) where C in {1, 2} and T > 0, or (T,) with T > 0
            - np.ndarray: Same shape constraints

    Example:
        >>> is_audio(torch.randn(2, 16000))
        True
        >>> is_audio(torch.randn(16000))
        True
        >>> is_audio(torch.randn(3, 16000))  # 3 channels not standard
        False
    """
    if isinstance(audio, torch.Tensor):
        if audio.ndim == 1:
            return audio.shape[0] > 0
        if audio.ndim == 2:
            c, t = audio.shape
            return c in (1, 2) and t > 0
        return False

    if isinstance(audio, np.ndarray):
        if audio.ndim == 1:
            return audio.shape[0] > 0
        if audio.ndim == 2:
            c, t = audio.shape
            return c in (1, 2) and t > 0
        return False

    return False


def is_audio_batch(audios: Any) -> bool:
    """
    Check if the input is a valid batch of audio waveforms.
    Corresponds to type ``AudioBatch``.

    Args:
        audios: Input to check.

    Returns:
        bool: True if valid audio batch:
            - torch.Tensor: Shape (B, C, T) where C in {1, 2}
            - np.ndarray: Shape (B, C, T) where C in {1, 2}
            - List[torch.Tensor] or List[np.ndarray]: Non-empty list of valid audios

    Example:
        >>> is_audio_batch(torch.randn(4, 2, 16000))
        True
        >>> is_audio_batch([torch.randn(1, 16000), torch.randn(1, 32000)])
        True
    """
    # 3D Tensor: (B, C, T)
    if isinstance(audios, torch.Tensor):
        if audios.ndim != 3:
            return False
        b, c, t = audios.shape
        return b > 0 and c in (1, 2) and t > 0

    # 3D NumPy: (B, C, T)
    if isinstance(audios, np.ndarray):
        if audios.ndim != 3:
            return False
        b, c, t = audios.shape
        return b > 0 and c in (1, 2) and t > 0

    # List of audio tensors/arrays
    if isinstance(audios, list) and len(audios) > 0:
        return all(is_audio(a) for a in audios)

    return False


# ----------------------------------- Loading / Saving --------------------------------------

def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = False,
) -> torch.Tensor:
    """
    Load an audio file as a waveform tensor.

    Uses ``torchaudio`` as the primary backend, with ``soundfile`` as fallback.
    The returned waveform is float32 in the range [-1.0, 1.0].

    Args:
        path: Path to audio file (.wav, .mp3, .flac, .ogg, etc.)
        sample_rate: If specified, resample to this rate. None keeps the original rate.
        mono: If True, downmix to mono by averaging channels.

    Returns:
        torch.Tensor: Waveform tensor of shape (C, T), float32 in [-1, 1].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ImportError: If neither torchaudio nor soundfile is available.

    Example:
        >>> waveform = load_audio("speech.wav", sample_rate=16000, mono=True)
        >>> waveform.shape
        torch.Size([1, 16000])  # 1 second of mono 16kHz audio
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, orig_sr = _load_audio_backend(str(path))

    # Resample if needed
    if sample_rate is not None and orig_sr != sample_rate:
        waveform = _resample(waveform, orig_sr, sample_rate)

    # Downmix to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sample_rate: int = 16000,
) -> None:
    """
    Save a waveform tensor to an audio file via torchaudio.

    The waveform is expected to be float32 in [-1, 1]. Values are clamped
    to that range. Output format is inferred by torchaudio from the path
    extension; any backend, codec, or I/O failure propagates to the caller.

    Args:
        waveform: Waveform tensor of shape (C, T) or (T,).
        path: Output file path. Format inferred from extension.
        sample_rate: Sample rate in Hz.

    Example:
        >>> waveform = torch.randn(1, 16000)
        >>> save_audio(waveform, "output.wav", sample_rate=16000)
    """
    import torchaudio

    path = Path(path)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.clamp(-1.0, 1.0).cpu().float()

    torchaudio.save(str(path), waveform, sample_rate)


# ----------------------------------- Conversions --------------------------------------

def audio_to_tensor(audio: AudioSingle) -> torch.Tensor:
    """
    Convert an audio waveform to a torch.Tensor of shape (C, T).

    1D inputs are promoted to (1, T) (mono).

    Args:
        audio: Audio waveform as Tensor or NumPy array.

    Returns:
        torch.Tensor: Shape (C, T), float32.

    Raises:
        TypeError: If input is not a Tensor or ndarray.

    Example:
        >>> audio_to_tensor(np.zeros(16000)).shape
        torch.Size([1, 16000])
    """
    if isinstance(audio, torch.Tensor):
        t = audio.float()
    elif isinstance(audio, np.ndarray):
        t = torch.from_numpy(audio).float()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(audio).__name__}")

    # Promote 1D to (1, T)
    if t.ndim == 1:
        t = t.unsqueeze(0)

    return t


def audio_to_numpy(audio: AudioSingle) -> np.ndarray:
    """
    Convert an audio waveform to a NumPy array of shape (C, T).

    1D inputs are promoted to (1, T) (mono).

    Args:
        audio: Audio waveform as Tensor or NumPy array.

    Returns:
        np.ndarray: Shape (C, T), float32.

    Raises:
        TypeError: If input is not a Tensor or ndarray.

    Example:
        >>> audio_to_numpy(torch.zeros(16000)).shape
        (1, 16000)
    """
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().float().numpy()
    elif isinstance(audio, np.ndarray):
        arr = audio.astype(np.float32)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(audio).__name__}")

    # Promote 1D to (1, T)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr


def convert_audio(
    waveform: torch.Tensor,
    from_rate: int,
    to_rate: int,
    to_channels: Optional[int] = None,
) -> torch.Tensor:
    """
    Resample and/or change the channel count of a waveform tensor.

    Follows the audiocraft convention for channel conversion:
        - Downmix to mono: average across channels
        - Upmix mono to stereo: repeat the channel

    Args:
        waveform: Input tensor of shape (C, T) or (B, C, T).
        from_rate: Original sample rate.
        to_rate: Target sample rate.
        to_channels: Target number of channels (1 or 2). None keeps original.

    Returns:
        torch.Tensor: Converted waveform, same number of dimensions as input.

    Example:
        >>> stereo = torch.randn(2, 32000)  # 2s stereo at 16kHz
        >>> mono_8k = convert_audio(stereo, from_rate=16000, to_rate=8000, to_channels=1)
        >>> mono_8k.shape
        torch.Size([1, 16000])
    """
    # Resample
    if from_rate != to_rate:
        waveform = _resample(waveform, from_rate, to_rate)

    # Channel conversion
    if to_channels is not None:
        waveform = _convert_channels(waveform, to_channels)

    return waveform


# ----------------------------------- Standardization --------------------------------------

def standardize_audio_batch(
    audios: Union[AudioSingle, AudioBatch],
    output_type: Literal['np', 'pt'] = 'pt',
) -> AudioBatch:
    """
    Standardize input audio(s) (single or batch) to the desired output format.

    Supports automatic promotion of single audio clips to batch format and preserves
    input structure: stacked input -> stacked output, list input -> list output.

    All 1D inputs are promoted to 2D (1, T) (mono) before processing.

    Args:
        audios: Input audio(s) in any supported format:
            - Single: torch.Tensor (C,T) or (T,), np.ndarray (C,T) or (T,)
            - Batch: torch.Tensor (B,C,T), np.ndarray (B,C,T)
            - List[torch.Tensor] or List[np.ndarray] (variable lengths allowed)
        output_type: Target format.
            - 'np': np.ndarray (B,C,T) if stacked, else List[np.ndarray]
            - 'pt': torch.Tensor (B,C,T) if stacked, else List[torch.Tensor]

    Returns:
        AudioBatch: Standardized batch in the requested format.

    Raises:
        ValueError: If input type is unsupported.

    Example:
        >>> # Single audio -> batch
        >>> clip = torch.randn(1, 16000)
        >>> batch = standardize_audio_batch(clip, output_type='pt')
        >>> batch.shape
        torch.Size([1, 1, 16000])

        >>> # Ragged list -> list output
        >>> clips = [torch.randn(1, 16000), torch.randn(1, 32000)]
        >>> batch = standardize_audio_batch(clips, output_type='pt')
        >>> len(batch)
        2
    """
    # --- Single tensor ---
    if isinstance(audios, torch.Tensor):
        # 1D: (T,) -> (1, 1, T)
        if audios.ndim == 1:
            audios = audios.unsqueeze(0).unsqueeze(0)
        # 2D: (C, T) -> (1, C, T)
        elif audios.ndim == 2:
            audios = audios.unsqueeze(0)
        # 3D: (B, C, T) -> (B, C, T)
        elif audios.ndim != 3:
            raise ValueError(
                f"expected audio tensor with 1-3 dims, got ndim={audios.ndim} "
                f"with shape {tuple(audios.shape)}"
            )
        if output_type == 'np':
            return audios.detach().cpu().float().numpy()
        return audios  # pt

    # --- Single ndarray ---
    if isinstance(audios, np.ndarray):
        if audios.ndim == 1:
            audios = audios[np.newaxis, np.newaxis, :]
        elif audios.ndim == 2:
            audios = audios[np.newaxis, :]
        elif audios.ndim != 3:
            raise ValueError(
                f"expected audio ndarray with 1-3 dims, got ndim={audios.ndim} "
                f"with shape {audios.shape}"
            )
        if output_type == 'pt':
            return torch.from_numpy(audios).float()
        return audios.astype(np.float32)  # np

    # --- List ---
    if isinstance(audios, list) and len(audios) > 0:
        # Normalize each element to 2D first
        if isinstance(audios[0], torch.Tensor):
            normalized = [a.unsqueeze(0) if a.ndim == 1 else a for a in audios]
            # Try to stack if all shapes match
            if all(a.shape == normalized[0].shape for a in normalized):
                stacked = torch.stack(normalized, dim=0)
                if output_type == 'np':
                    return stacked.detach().cpu().float().numpy()
                return stacked  # pt
            # Ragged: return as list
            if output_type == 'np':
                return [a.detach().cpu().float().numpy() for a in normalized]
            return normalized  # pt

        if isinstance(audios[0], np.ndarray):
            normalized = [a[np.newaxis, :] if a.ndim == 1 else a for a in audios]
            if all(a.shape == normalized[0].shape for a in normalized):
                stacked = np.stack(normalized, axis=0)
                if output_type == 'pt':
                    return torch.from_numpy(stacked).float()
                return stacked.astype(np.float32)  # np
            if output_type == 'pt':
                return [torch.from_numpy(a).float() for a in normalized]
            return [a.astype(np.float32) for a in normalized]  # np

    raise ValueError(f"Unsupported audio input type: {type(audios)}")


# ----------------------------------- Hashing --------------------------------------

def hash_audio(audio: torch.Tensor, max_samples: int = 4096) -> str:
    """
    Generate a stable hash string for an audio waveform tensor.

    Subsamples long audio for efficiency. The hash is computed on raw float bytes
    after deterministic rounding, following the same pattern as ``hash_tensor``.

    Args:
        audio: Waveform tensor of shape (C, T) or (T,).
        max_samples: Maximum number of time-domain samples to hash.
            Longer audio is uniformly subsampled.

    Returns:
        str: MD5 hash hex string.

    Example:
        >>> hash_audio(torch.zeros(1, 16000))
        'a1...'  # deterministic for identical inputs
    """
    flat = audio.detach().flatten()
    n = flat.numel()

    # Subsample for efficiency
    if n > max_samples:
        step = n // max_samples
        flat = flat[::step][:max_samples]

    # Quantize to int16 (matches WAV precision, eliminates float rounding issues)
    quantized = (flat.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    return hashlib.md5(quantized.cpu().numpy().tobytes()).hexdigest()


def hash_audio_list(audios: List[torch.Tensor], max_samples: int = 4096) -> str:
    """
    Generate a combined hash for a list of audio waveforms.

    Args:
        audios: List of waveform tensors.
        max_samples: Maximum samples to hash per clip.

    Returns:
        str: Combined MD5 hash hex string.

    Example:
        >>> hash_audio_list([torch.zeros(1, 16000), torch.ones(1, 16000)])
        'b2...'
    """
    hasher = hashlib.md5()
    for audio in audios:
        hasher.update(hash_audio(audio, max_samples=max_samples).encode())
    return hasher.hexdigest()


# ----------------------------------- Internal Helpers --------------------------------------

def _load_audio_backend(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio using the best available backend.

    Priority: torchaudio > soundfile + torch.

    Returns:
        Tuple of (waveform (C, T) float32, sample_rate).
    """
    # Try torchaudio first (handles most formats including mp3)
    try:
        import torchaudio
    except ImportError:
        pass
    else:
        waveform, sr = torchaudio.load(path)
        return waveform, sr

    # Fallback to soundfile (handles wav, flac, ogg)
    try:
        import soundfile as sf
    except ImportError:
        pass
    else:
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        # soundfile returns (T, C), convert to (C, T)
        waveform = torch.from_numpy(data.T).float()
        return waveform, sr

    # Last resort: stdlib wave module (WAV only, always available)
    import wave
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:  # int16
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:  # int32
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes. Only 16-bit and 32-bit WAV supported.")

    # De-interleave channels: (T*C,) -> (C, T)
    samples = samples.reshape(-1, n_channels).T
    return torch.from_numpy(samples).float(), sr


def _resample(waveform: torch.Tensor, from_rate: int, to_rate: int) -> torch.Tensor:
    """
    Resample waveform via torchaudio.functional.resample.

    Any dtype/device/rate error from torchaudio propagates to the caller.
    """
    if from_rate == to_rate:
        return waveform

    import torchaudio.functional as F

    return F.resample(waveform, from_rate, to_rate)


def _convert_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    """
    Convert waveform channel count, following audiocraft convention.

    Args:
        waveform: (C, T) or (B, C, T)
        target_channels: 1 (mono) or 2 (stereo)
    """
    # Determine channel dim position
    if waveform.ndim == 2:
        ch_dim = 0
    elif waveform.ndim == 3:
        ch_dim = 1
    else:
        return waveform

    current = waveform.shape[ch_dim]
    if current == target_channels:
        return waveform

    if target_channels == 1:
        # Downmix to mono: average across channels
        return waveform.mean(dim=ch_dim, keepdim=True)

    if target_channels == 2 and current == 1:
        # Upmix mono to stereo: repeat
        return waveform.repeat_interleave(2, dim=ch_dim)

    raise ValueError(
        f"Cannot convert {current} channels to {target_channels}. "
        f"Supported conversions: stereo->mono, mono->stereo."
    )
