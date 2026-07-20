"""
Optional speaker diarization ("who's speaking when") via SpeechBrain's
ECAPA-TDNN speaker-embedding model.

Why SpeechBrain instead of pyannote (the more commonly recommended library):
pyannote's pretrained pipelines are gated on Hugging Face — using them
requires the end user to create an HF account, manually accept the model's
license on its page, generate an access token, and pass that token in. For a
pip-installed tool, that's a meaningfully bigger ask than a plain
`pip install`. SpeechBrain's `speechbrain/spkrec-ecapa-voxceleb` model is
Apache-2.0 licensed and freely downloadable — no account, no license
click-through, no token. The tradeoff: this isn't a dedicated end-to-end
diarization pipeline like pyannote's — it's speaker *embeddings*, which this
module turns into diarization itself via sliding-window embedding +
clustering. That's a real accuracy tradeoff, not just a licensing one — see
ACCURACY LIMITATIONS below. It's also a genuinely heavy dependency either
way (PyTorch + torchaudio + speechbrain), which is why this whole feature is
opt-in rather than always installed.

ACCURACY LIMITATIONS — read this before trusting the output:
    - No dedicated voice-activity-detection model: silence is skipped with a
      simple energy (RMS) threshold, not a real VAD, so quiet speech can be
      mistaken for silence and loud noise can be mistaken for speech.
    - Overlapping speech (people talking over each other) isn't modeled —
      a window containing two people gets assigned to whichever one speaker
      the embedding looks closest to, not "both."
    - Similar-sounding voices are the most common failure case — clustering
      can merge two real speakers into one, or split one speaker into two if
      their voice varies a lot (shouting vs. quiet, close-mic vs. far-mic).
    - Without a known speaker count, the automatic cluster-count estimate is
      a rough distance-threshold heuristic and can over- or under-count
      speakers. Providing the expected count when you know it is
      meaningfully more reliable than letting it guess.
    - Speaker identity is assigned per *transcript segment* as a whole (via
      maximum time-overlap with detected speaker turns), not split at the
      exact word where a speaker change happens mid-segment — so a sentence
      spanning a speaker change gets attributed to whichever speaker covers
      more of it.
    - This is an estimate to help organize a transcript, not a verified,
      ground-truth speaker log.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np

WINDOW_SEC = 1.5
HOP_SEC = 0.75
MIN_TAIL_SEC = 0.3
SILENCE_RMS_THRESHOLD = 0.01
DEFAULT_DISTANCE_THRESHOLD = 0.6  # used only when num_speakers isn't given

_ENCODER = None  # lazy singleton — loaded once per process, reused after that


def _decode_to_wav_16k_mono(input_path: str) -> str:
    """ffmpeg-decode any audio/video file to a standardized 16kHz mono
    16-bit PCM wav, so format support depends on ffmpeg (already a hard
    requirement of this project) rather than on torchaudio's own format
    support / backend churn (recent torchaudio versions need a separate
    'torchcodec' package just to load a file — not worth depending on when
    ffmpeg can already do this decode step directly)."""
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "diarize_input.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Couldn't decode audio for diarization: {result.stderr[-2000:]}")
    return out_path


def _read_wav_as_tensor(wav_path: str):
    """Read a 16-bit PCM mono wav (as produced by _decode_to_wav_16k_mono)
    into a 1-D float32 torch tensor in [-1, 1], plus its sample rate."""
    import torch

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError(f"Expected 16-bit PCM wav, got sample width {sampwidth} bytes")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(data), framerate


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "Speaker diarization needs the optional 'speechbrain', 'torch', "
                "and 'scikit-learn' packages. Install with: "
                "pip install speechbrain torch scikit-learn"
                " (or: pip install mtt-transcriber[diarization])"
            ) from e
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(tempfile.gettempdir(), "speechbrain_ecapa_voxceleb"),
        )
    return _ENCODER


def diarize_audio(input_path: str, num_speakers: int | None = None):
    """Returns a list of {"start": float, "end": float, "speaker": "SPEAKER_00"}
    turns for the given audio/video file.

    `num_speakers`, if given, is used directly for clustering and is
    meaningfully more reliable than leaving it as None, which falls back to
    an automatic (rougher) speaker-count estimate. See module docstring for
    accuracy limitations.
    """
    import torch
    from sklearn.cluster import AgglomerativeClustering

    encoder = _get_encoder()  # do this first: surfaces a clear ImportError early if deps are missing

    wav_path = _decode_to_wav_16k_mono(input_path)
    waveform, sr = _read_wav_as_tensor(wav_path)
    total_samples = waveform.shape[0]

    win = int(WINDOW_SEC * sr)
    hop = int(HOP_SEC * sr)
    min_tail = int(MIN_TAIL_SEC * sr)

    windows = []  # (start_sec, end_sec, chunk_tensor)
    i = 0
    while i < total_samples:
        chunk = waveform[i:i + win]
        if chunk.shape[0] < min_tail:
            break
        rms = float(torch.sqrt(torch.mean(chunk ** 2)))
        if rms >= SILENCE_RMS_THRESHOLD:
            windows.append((i / sr, min((i + win) / sr, total_samples / sr), chunk))
        i += hop

    if not windows:
        return []

    embeddings = []
    with torch.no_grad():
        for _start, _end, chunk in windows:
            emb = encoder.encode_batch(chunk.unsqueeze(0)).squeeze().cpu().numpy()
            embeddings.append(emb)
    embeddings = np.stack(embeddings)

    if len(windows) == 1:
        labels = [0]
    elif num_speakers:
        clustering = AgglomerativeClustering(
            n_clusters=min(num_speakers, len(windows)), metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(embeddings)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
            metric="cosine", linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

    # Merge consecutive/overlapping windows with the same speaker label into
    # continuous turns, rather than leaving lots of small overlapping windows.
    turns = []
    for (start, end, _chunk), label in zip(windows, labels):
        speaker = f"SPEAKER_{int(label):02d}"
        if turns and turns[-1]["speaker"] == speaker and start <= turns[-1]["end"] + HOP_SEC + 0.05:
            turns[-1]["end"] = end
        else:
            turns.append({"start": start, "end": end, "speaker": speaker})
    return turns


def assign_speakers(segments, turns):
    """Tag each transcript segment with whichever speaker turn overlaps it
    the most (by total time overlap). Returns a NEW list — doesn't mutate
    the input, so it's safe to call on an st.cache_data-returned object."""
    if not turns:
        return [dict(s, speaker=None) for s in segments]
    tagged = []
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn["speaker"]
        tagged.append(dict(seg, speaker=best_speaker or turns[0]["speaker"]))
    return tagged


def speaker_labels_in(segments):
    """Unique raw speaker labels (e.g. SPEAKER_00) present, in first-appearance order."""
    seen = []
    for s in segments:
        sp = s.get("speaker")
        if sp and sp not in seen:
            seen.append(sp)
    return seen


def apply_speaker_names(segments, name_map: dict):
    """Return a NEW list of segments with a `speaker_display` field: a
    user-friendly name substituted via name_map (e.g. {"SPEAKER_00": "Alice"}),
    falling back to the raw label if not present/blank. Doesn't touch `text`."""
    out = []
    for seg in segments:
        sp = seg.get("speaker")
        display = (name_map.get(sp) or sp) if sp else None
        out.append(dict(seg, speaker_display=display))
    return out


def speaker_prefix(seg) -> str:
    """`"[Name] "` if this segment has a resolved speaker display name, else ""."""
    name = seg.get("speaker_display")
    return f"[{name}] " if name else ""
