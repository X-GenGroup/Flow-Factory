# src/flow_factory/rewards/pronunciation.py
"""Pronunciation reward: ASR-based CER between generated audio and target utterance.

Uses SenseVoice (FunASR) to transcribe the generated audio, extracts the target
utterance from the prompt (text enclosed in Chinese quotes ``\u201c...\u201d``),
and computes the Character Error Rate (CER).  The reward is ``1 - CER`` so that
lower CER → higher reward (range [0, 1]).

If no target utterance is found in the prompt, the reward falls back to 0.5
(neutral) so the sample does not skew the advantage.

Configuration (RewardArguments.extra_kwargs):
    asr_model_path: path to SenseVoice model directory
    language: language hint for ASR (default "Chinese")
    neutral_reward: reward when no target utterance is available (default 0.5)
"""
import math
import re
import tempfile
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from .registry import register_reward_model
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_QUOTE_RE = re.compile(r'[\u201c"](.*?)[\u201d"]')


def _calc_cer(target: str, asr_text: str) -> float:
    """Character Error Rate for Chinese characters."""
    import difflib

    t = [c for c in target if _HAN_RE.match(c)]
    a = [c for c in asr_text if _HAN_RE.match(c)]
    if not t:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, t)
    matches = sum(b.size for b in sm.get_matching_blocks())
    edits = len(t) - matches
    return edits / len(t)


def _extract_utterance(prompt: str) -> Optional[str]:
    """Extract target utterance from prompt (text in Chinese quotes \u201c...\u201d)."""
    m = _QUOTE_RE.search(prompt)
    return m.group(1) if m else None


@register_reward_model("pronunciation")
class PronunciationRewardModel(PointwiseRewardModel):
    """Reward = 1 - CER between ASR transcription of generated audio and target utterance."""

    required_fields: Tuple[str, ...] = ("prompt", "audio")
    use_tensor_inputs: bool = False  # audio arrives as np.ndarray (C, T) float32

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        ek = config.extra_kwargs or {}
        self.asr_model_path = ek.get(
            "asr_model_path",
            "/inspire/sj-ssd3/project/embodied-multimodality-ascend/public/xuehaiwei/ltx/asr_models/iic/SenseVoiceSmall",
        )
        self.language = ek.get("language", "Chinese")
        self.neutral_reward = float(ek.get("neutral_reward", 0.5))
        self._asr_model = None

    def _load_asr(self):
        if self._asr_model is not None:
            return self._asr_model
        try:
            from funasr import AutoModel

            self._asr_model = AutoModel(
                model=self.asr_model_path,
                trust_remote_code=True,
                device=str(self.device),
                disable_update=True,
            )
            logger.info(f"[pronunciation] SenseVoice ASR loaded from {self.asr_model_path}")
        except Exception as e:
            logger.error(f"[pronunciation] Failed to load SenseVoice: {e}")
            self._asr_model = False  # mark as failed
        return self._asr_model

    def _transcribe(self, audio_np: np.ndarray, sample_rate: int = 24000) -> str:
        """Transcribe audio waveform (shape (C, T) or (T,), float32 [-1,1]) via SenseVoice."""
        asr = self._load_asr()
        if not asr:
            return ""

        # SenseVoice expects 16kHz mono. audio_np is (C, T) float32 in [-1, 1].
        wav = audio_np
        if wav.ndim == 2:
            wav = wav[0]  # take first channel
        if sample_rate != 16000:
            # LTX-2 vocoder outputs 24kHz; resample to 16kHz. Writing a 24kHz
            # waveform with a 16kHz header makes ASR hear 1.5x-slowed audio and
            # saturates CER at ~1 (reward ~0).
            from scipy.signal import resample_poly

            g = math.gcd(16000, int(sample_rate))
            wav = resample_poly(wav, 16000 // g, int(sample_rate) // g)
        wav = (wav * 32767).clip(-32768, 32767).astype(np.int16)

        # Write to temp wav file (FunASR reads from path)
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            import soundfile as sf

            sf.write(tmp_wav, wav, 16000, subtype="PCM_16")
            res = asr.generate(
                input=tmp_wav,
                language=self.language,
                use_itn=True,
                batch_size_s=0,
            )
            if res and isinstance(res, list):
                text = res[0].get("text", "")
            elif res and isinstance(res, dict):
                text = res.get("text", "")
            else:
                text = ""
            # Clean SenseVoice special tokens
            text = re.sub(r"<\|[^|]+\|>", "", text).strip()
            return text
        except Exception as e:
            logger.warning(f"[pronunciation] ASR transcription failed: {e}")
            return ""
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        audio: Optional[List] = None,
        audio_sample_rate: Optional[List] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if audio is None:
            raise ValueError("PronunciationRewardModel requires `audio`.")

        rewards = []
        cer_list = []
        asr_texts = []
        targets = []

        for i, aud in enumerate(audio):
            target = _extract_utterance(prompt[i])
            targets.append(target)

            if target is None:
                rewards.append(self.neutral_reward)
                cer_list.append(None)
                asr_texts.append("")
                continue

            # audio can be np.ndarray or torch.Tensor
            if isinstance(aud, torch.Tensor):
                aud = aud.float().cpu().numpy()
            if isinstance(aud, np.ndarray):
                # Per-sample source rate (LTX-2 vocoder: 24000); default covers
                # adapters that do not report audio_sample_rate.
                rate = audio_sample_rate[i] if audio_sample_rate is not None else 24000
                asr_text = self._transcribe(aud, int(rate))
            else:
                asr_text = ""

            cer = _calc_cer(target, asr_text)
            rewards.append(1.0 - cer)
            cer_list.append(cer)
            asr_texts.append(asr_text)

        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),
            extra_info={
                "cer": cer_list,
                "asr_text": asr_texts,
                "target": targets,
            },
        )
