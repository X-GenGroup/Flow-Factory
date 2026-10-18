# MiniMax H3 Ref2VA manifest fixtures

See the canonical [Dataset Guide](../../guidance/datasets.md#ref2va-minimax-h3-ref2va) for the
complete schema, preprocessing/cache flow, and training semantics.

Each JSONL row contains a prompt and a non-empty ordered `references` array. Array order is
semantically significant, is preserved during canonicalization, and participates in sample
identity.

Supported entries:

- `image`: `type` and dataset-relative `path`;
- `video`: `type`, dataset-relative `path`, optional finite positive `fps`, and optional
  dataset-relative `audio_path`;
- `audio`: `type`, dataset-relative `path`, and optional finite positive `sample_rate`.

At least one image or video is required; audio-only manifests are invalid. A video `sample_rate`
is valid only when that video also supplies `audio_path`. Manifest `fps` and `sample_rate`
overrides take precedence over decoded metadata where supported.

PyAV >=17.0.0 is required for reliable video/audio decoding. It preserves video frames and FPS,
plus embedded or separately referenced audio and its sample rate. Only encoded tensors/layout and
the canonical manifest enter the Arrow cache; upstream reference objects are transient.

These fixtures are intentionally tiny schema examples. The associated configuration has not
loaded the 61 GB checkpoint and does not establish generated-media, numerical, memory, training,
or reward parity.
