"""
Local Media Transcriber + Synced / Overlaid Captions (faster-whisper)
======================================================================

Upload one or more audio/video files (instead of a YouTube URL) and get:
    - Batch mode: upload several files at once, each processed independently
      under its own tab, named after the source file. Sidebar settings
      (model size, translation) apply to all tabs; each tab has its own
      transcribe/translate/export buttons and results.
    - Native playback (HTML5 <video> or <audio>)
    - A timestamped transcript (faster-whisper), synced to playback using
      the browser's native `timeupdate` event — no external API/polling
      needed, since we own the media element directly.
    - For video: a subtitle-style caption overlay burned visually onto the
      player as it plays. For audio: a large "lyrics-style" caption line
      that updates in sync, since there's no video to overlay onto.
    - Optional translation to another language (Google Translate via the
      free `deep-translator` package), with Original / Translated / Both
      display modes — same as the YouTube version.
    - Export with captions burned in as a real downloadable .mp4:
        * uploaded video -> hardcodes captions directly onto your video
        * uploaded audio -> synthesizes a simple captioned "lyric video"
          (solid background color, optional waveform) using your audio

Run with:
    streamlit run media_file_transcriber.py

Requirements (pip install):
    streamlit
    faster-whisper
    deep-translator
    edge-tts

Supported formats:
    Audio: .mp3 .wav .m4a .aac .flac .ogg .opus .wma
    Video: .mp4 .m4v .webm .mov .mkv .avi .flv
    (faster-whisper/ffmpeg can decode all of these for transcription and
    export regardless of browser support; a few formats — .wma, .mov,
    .mkv, .avi, .flv — commonly don't play back natively in the browser's
    preview player, which the app will flag when relevant.)

System requirement:
    ffmpeg on PATH, built with libass support (needed for the `subtitles`
    filter). Check with: ffmpeg -filters | grep subtitles

Notes:
    - faster-whisper decodes audio directly from video containers, so no
      separate audio-extraction step is needed even for .mp4 uploads.
    - Streamlit's default upload limit is 200MB (configurable via
      `maxUploadSize` in .streamlit/config.toml). Larger files also take
      longer to embed for playback, since the file is inlined as a base64
      data URI for the custom player/overlay component.
"""

import os
import re
import json
import hashlib
import tempfile
import subprocess
import base64

import streamlit as st
import streamlit.components.v1 as components


UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "media_transcriber_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def saved_path(file_id: str, suffix: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{file_id}{suffix}")


def ensure_saved(file_id: str, data: bytes, suffix: str) -> str:
    path = saved_path(file_id, suffix)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@st.cache_data(show_spinner=False)
def transcribe(file_id: str, suffix: str, model_size: str):
    path = saved_path(file_id, suffix)
    model = load_model(model_size)
    segments, _info = model.transcribe(path, beam_size=5, vad_filter=True)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]


# (mime type, is_video, plays natively in most browsers)
EXT_INFO = {
    # Audio
    ".mp3":  ("audio/mpeg",       False, True),
    ".wav":  ("audio/wav",        False, True),
    ".m4a":  ("audio/mp4",        False, True),
    ".aac":  ("audio/aac",        False, True),
    ".flac": ("audio/flac",       False, True),
    ".ogg":  ("audio/ogg",        False, True),
    ".opus": ("audio/ogg",        False, True),
    ".wma":  ("audio/x-ms-wma",   False, False),
    # Video
    ".mp4":  ("video/mp4",        True,  True),
    ".m4v":  ("video/mp4",        True,  True),
    ".webm": ("video/webm",       True,  True),
    ".mov":  ("video/quicktime",  True,  False),
    ".mkv":  ("video/x-matroska", True,  False),
    ".avi":  ("video/x-msvideo",  True,  False),
    ".flv":  ("video/x-flv",      True,  False),
}

SUPPORTED_EXTS = [ext.lstrip(".") for ext in EXT_INFO]


LANGUAGES = {
    "English": "en",
    "Japanese": "ja",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Korean": "ko",
    "Hindi": "hi",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Italian": "it",
    "Vietnamese": "vi",
    "Thai": "th",
}


@st.cache_data(show_spinner=False)
def translate_segments(file_id: str, suffix: str, model_size: str, target_lang_code: str):
    from deep_translator import GoogleTranslator

    segments = transcribe(file_id, suffix, model_size)
    texts = [s["text"] if s["text"] else " " for s in segments]

    translator = GoogleTranslator(source="auto", target=target_lang_code)
    translated_texts = []
    chunk_size = 50
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        translated_texts.extend(translator.translate_batch(chunk))

    return [
        {"start": s["start"], "end": s["end"], "text": (t or "").strip()}
        for s, t in zip(segments, translated_texts)
    ]


def fmt_time(t: float) -> str:
    t = int(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def to_srt_timestamp(t: float) -> str:
    ms_total = int(round(t * 1000))
    h, ms_total = divmod(ms_total, 3600000)
    m, ms_total = divmod(ms_total, 60000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{to_srt_timestamp(seg['start'])} --> {to_srt_timestamp(seg['end'])}")
        lines.append(seg["text"] if seg["text"] else " ")
        lines.append("")
    return "\n".join(lines)


def _subtitles_filter(srt_path: str, font_size: int = 24) -> str:
    escaped_path = srt_path.replace("\\", "/").replace(":", "\\:")
    style = (
        f"FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF&,"
        "OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0,MarginV=40"
    )
    return f"subtitles='{escaped_path}':force_style='{style}'"


@st.cache_data(show_spinner=False)
def render_video_with_captions(file_id: str, suffix: str, srt_text: str) -> bytes:
    """Uploaded VIDEO: hardcode the captions directly onto it."""
    video_path = saved_path(file_id, suffix)
    out_dir = tempfile.mkdtemp()
    srt_path = os.path.join(out_dir, "captions.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    out_path = os.path.join(out_dir, f"{file_id}_captioned.mp4")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", _subtitles_filter(srt_path),
        "-c:a", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    with open(out_path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def render_lyric_video(
    file_id: str, suffix: str, srt_text: str,
    width: int, height: int, bg_color: str, waveform: bool,
) -> bytes:
    """Uploaded AUDIO: synthesize a simple captioned 'lyric video'."""
    audio_path = saved_path(file_id, suffix)
    out_dir = tempfile.mkdtemp()
    srt_path = os.path.join(out_dir, "captions.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    out_path = os.path.join(out_dir, f"{file_id}_lyric_video.mp4")
    sub_filter = _subtitles_filter(srt_path, font_size=max(20, height // 25))

    if waveform:
        wave_h = int(height * 0.28)
        filter_complex = (
            f"[1:a]showwaves=s={width}x{wave_h}:mode=cline:colors=white[wave];"
            f"[0:v][wave]overlay=0:H-h-40[bg];"
            f"[bg]{sub_filter}[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r=25",
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[out]", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r=25",
            "-i", audio_path,
            "-vf", sub_filter,
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            out_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    with open(out_path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def extract_original_audio_mp3(file_id: str, suffix: str, is_video: bool) -> bytes:
    """Just the audio track — extracts it if the upload was a video, or passes
    an already-audio upload through unchanged (re-muxed to mp3 for consistency)."""
    src_path = saved_path(file_id, suffix)
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"{file_id}_original.mp3")
    cmd = ["ffmpeg", "-y", "-i", src_path]
    if is_video:
        cmd += ["-vn"]
    cmd += ["-c:a", "libmp3lame", "-q:a", "2", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    with open(out_path, "rb") as f:
        return f.read()


EDGE_VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "ko": "ko-KR-SunHiNeural",
    "hi": "hi-IN-SwaraNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "it": "it-IT-ElsaNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
}

MAX_RATE_SPEEDUP_PCT = 50  # cap how much we'll speed up a line to make it fit


async def _edge_tts_save(text: str, voice: str, rate_str: str, out_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(out_path)


def edge_tts_save(text: str, voice: str, rate_str: str, out_path: str):
    import asyncio
    asyncio.run(_edge_tts_save(text, voice, rate_str, out_path))


def probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


@st.cache_data(show_spinner=False)
def synthesize_dubbed_audio(file_id: str, lang_code: str, segments_json: str) -> bytes:
    """Rough dubbed audio: TTS each translated line (via edge-tts), speeding up
    lines that run long for their time slot, then delay-and-mix at timestamp.

    This narrows the overlap problem but doesn't eliminate it — a line is only
    sped up by up to MAX_RATE_SPEEDUP_PCT to stay intelligible, so lines that
    would need a bigger speedup than that to fit will still overlap somewhat.
    """
    segments = json.loads(segments_json)
    voice = EDGE_VOICE_MAP.get(lang_code, "en-US-AriaNeural")
    work_dir = tempfile.mkdtemp()
    inputs, filter_parts = [], []
    idx = 0
    n = len(segments)

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg["start"]
        if i + 1 < n:
            available = max(0.4, segments[i + 1]["start"] - start)
        else:
            available = max(3.0, seg["end"] - start)

        base_path = os.path.join(work_dir, f"seg_{idx}_base.mp3")
        try:
            edge_tts_save(text, voice, "+0%", base_path)
        except Exception:
            continue

        final_path = base_path
        duration = probe_duration(base_path)
        if duration > available > 0:
            ratio = duration / available
            rate_pct = min(MAX_RATE_SPEEDUP_PCT, max(0, int(round((ratio - 1) * 100))))
            if rate_pct > 0:
                sped_path = os.path.join(work_dir, f"seg_{idx}_r{rate_pct}.mp3")
                try:
                    edge_tts_save(text, voice, f"+{rate_pct}%", sped_path)
                    final_path = sped_path
                except Exception:
                    pass

        delay_ms = max(0, int(start * 1000))
        inputs.append(final_path)
        filter_parts.append(f"[{idx}:a]adelay={delay_ms}:all=1[a{idx}]")
        idx += 1

    if idx == 0:
        raise RuntimeError("No translated text available to synthesize.")

    mix_labels = "".join(f"[a{i}]" for i in range(idx))
    filter_complex = ";".join(filter_parts) + f";{mix_labels}amix=inputs={idx}:normalize=0[mixed]"

    out_path = os.path.join(work_dir, f"{file_id}_dubbed.mp3")
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", filter_complex, "-map", "[mixed]", "-c:a", "libmp3lame", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    with open(out_path, "rb") as f:
        return f.read()


def caption_suffix(enable_translation: bool, translated, display_mode: str, lang_code: str) -> str:
    """Filename suffix reflecting which caption language(s) are baked into an export."""
    if not (enable_translation and translated):
        return ""
    if display_mode == "Translated only":
        return f"_{lang_code}"
    if display_mode == "Both":
        return f"_orig+{lang_code}"
    return ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def process_one_file(uploaded_file, model_size, enable_translation, target_lang_label, display_mode):
    """Runs the full single-file pipeline (transcribe/translate/preview/export)
    for one uploaded file. Called once per tab in batch mode."""
    data = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    if suffix not in EXT_INFO:
        st.error(f"Unsupported file type: {suffix}")
        return

    mime, is_video, preview_supported = EXT_INFO[suffix]
    fid = file_hash(data)
    ensure_saved(fid, data, suffix)

    raw_stem = os.path.splitext(uploaded_file.name)[0]
    base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_stem).strip("_") or fid

    if not preview_supported:
        st.info(
            f"Heads up: `{suffix}` files often don't play back natively in the browser, "
            "so the preview player below may not work. Transcription, translation, and "
            "MP4 export all work regardless."
        )

    size_mb = len(data) / 1e6
    if size_mb > 100:
        st.warning(
            f"This file is {size_mb:.0f} MB — embedding it for playback may be slow, "
            "since it's inlined as base64. Transcription/translation/export still work fine."
        )

    transcribe_key = f"transcript_{fid}_{model_size}"
    transcribe_clicked = st.button("Transcribe", type="primary", key=f"transcribe_btn_{fid}")
    if transcribe_clicked:
        with st.spinner("Transcribing..."):
            try:
                st.session_state[transcribe_key] = transcribe(fid, suffix, model_size)
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    transcript = st.session_state.get(transcribe_key)

    translated = None
    if transcript and enable_translation:
        target_lang_code = LANGUAGES[target_lang_label]
        with st.spinner(f"Translating to {target_lang_label}..."):
            try:
                translated = translate_segments(fid, suffix, model_size, target_lang_code)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                translated = None

    if transcript and enable_translation and translated:
        if display_mode == "Translated only":
            display_segments = translated
        elif display_mode == "Both":
            display_segments = [
                {"start": o["start"], "end": o["end"], "text": f"{o['text']}\n{t['text']}"}
                for o, t in zip(transcript, translated)
            ]
        else:
            display_segments = transcript
    else:
        display_segments = transcript

    show_overlay = st.checkbox(
        "Show caption overlay" if is_video else "Show synced caption line",
        value=True,
        key=f"show_overlay_{fid}",
    )
    segments_json = json.dumps(display_segments or [])
    overlay_enabled_js = "true" if show_overlay else "false"

    b64 = base64.b64encode(data).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"

    if is_video:
        player_block = f"""
        <div id="player-wrap" style="position:relative; width:100%; max-width:720px; aspect-ratio:16/9; background:#000; border-radius:8px; overflow:hidden;">
          <video id="mediaEl" controls style="width:100%; height:100%; display:block;" src="{data_uri}"></video>
          <div id="caption-overlay" style="
              position:absolute; left:50%; bottom:6%; transform:translateX(-50%);
              max-width:88%; text-align:center; pointer-events:none;
              background: rgba(0,0,0,0.7); color:#fff; padding:6px 14px;
              border-radius:6px; font-family:sans-serif; font-size:1.05em;
              line-height:1.35; white-space:pre-line; z-index:5; display:none;"></div>
        </div>
        """
    else:
        player_block = f"""
        <div id="player-wrap" style="width:100%; max-width:720px;">
          <audio id="mediaEl" controls style="width:100%;" src="{data_uri}"></audio>
          <div id="caption-overlay" style="
              margin-top: 16px; min-height: 70px; align-items:center; justify-content:center;
              text-align:center; background:#111; color:#fff; padding:14px 18px;
              border-radius:8px; font-family:sans-serif; font-size:1.3em;
              line-height:1.4; white-space:pre-line; display:none;"></div>
        </div>
        """

    html_code = f"""
    <div id="app">
      {player_block}
      <div id="transcript-box" style="
          height: 300px; overflow-y: auto; margin-top: 12px;
          border: 1px solid #444; border-radius: 8px; padding: 10px;
          font-family: sans-serif; font-size: 14px; background: #111; color: #eee; white-space:pre-line;">
        <div id="transcript-inner"></div>
      </div>
    </div>

    <script>
      const segments = {segments_json};
      const overlayEnabled = {overlay_enabled_js};
      const isVideo = {"true" if is_video else "false"};
      const mediaEl = document.getElementById('mediaEl');
      let currentIdx = -1;

      function formatTime(t) {{
        t = Math.floor(t);
        const h = Math.floor(t/3600), m = Math.floor((t%3600)/60), s = t%60;
        const mm = h ? String(m).padStart(2,'0') : m;
        const ss = String(s).padStart(2,'0');
        return h ? `${{h}}:${{mm}}:${{ss}}` : `${{mm}}:${{ss}}`;
      }}

      function escapeHtml(str) {{
        const d = document.createElement('div');
        d.innerText = str;
        return d.innerHTML;
      }}

      function seekTo(t) {{
        mediaEl.currentTime = t;
        mediaEl.play();
      }}

      function renderTranscript() {{
        const inner = document.getElementById('transcript-inner');
        if (!segments.length) {{
          inner.innerHTML = "<em>No transcript yet. Click 'Transcribe' in the app.</em>";
          return;
        }}
        inner.innerHTML = segments.map((seg, i) => `
          <div id="seg-${{i}}"
               onclick="seekTo(${{seg.start}})"
               style="padding:4px 6px; cursor:pointer; border-radius:4px; margin-bottom:2px;">
            <span style="color:#8ab4f8; margin-right:8px;">${{formatTime(seg.start)}}</span>${{escapeHtml(seg.text)}}
          </div>
        `).join('');
      }}

      function updateHighlight() {{
        if (!segments.length) return;
        const t = mediaEl.currentTime;
        let idx = -1;
        for (let i = 0; i < segments.length; i++) {{
          if (t >= segments[i].start && t < segments[i].end) {{ idx = i; break; }}
        }}
        if (idx !== currentIdx) {{
          if (currentIdx >= 0) {{
            const prevEl = document.getElementById('seg-' + currentIdx);
            if (prevEl) prevEl.style.background = 'transparent';
          }}
          const overlay = document.getElementById('caption-overlay');
          const shownDisplay = isVideo ? 'block' : 'flex';
          if (idx >= 0) {{
            const el = document.getElementById('seg-' + idx);
            if (el) {{
              el.style.background = '#2b3a55';
              el.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
            }}
            overlay.textContent = segments[idx].text;
            overlay.style.display = overlayEnabled ? shownDisplay : 'none';
          }} else {{
            overlay.textContent = '';
            overlay.style.display = 'none';
          }}
          currentIdx = idx;
        }}
      }}

      mediaEl.addEventListener('timeupdate', updateHighlight);
      renderTranscript();
    </script>
    """

    components.html(html_code, height=760 if is_video else 560, scrolling=False)

    if transcript:
        full_text = "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in transcript)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download original transcript (.txt)",
                data=full_text,
                file_name=f"{base_name}_transcript_orig.txt",
                mime="text/plain",
                key=f"dl_transcript_orig_{fid}",
            )
        if translated:
            translated_text = "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in translated)
            lang_code = LANGUAGES[target_lang_label]
            with dl_col2:
                st.download_button(
                    f"Download {target_lang_label} transcript (.txt)",
                    data=translated_text,
                    file_name=f"{base_name}_transcript_{lang_code}.txt",
                    mime="text/plain",
                    key=f"dl_transcript_translated_{fid}_{lang_code}",
                )

        st.divider()
        st.subheader("🔊 Download audio only")
        st.caption("No video, just an audio file — more than the plain-text transcript, less than a full video.")

        aud_col1, aud_col2 = st.columns(2)

        with aud_col1:
            if st.button("Prepare original audio (.mp3)", key=f"prep_orig_audio_btn_{fid}"):
                with st.spinner("Extracting original audio..."):
                    try:
                        st.session_state[f"orig_audio_{fid}"] = extract_original_audio_mp3(fid, suffix, is_video)
                    except Exception as e:
                        st.error(f"Audio extraction failed: {e}")
            orig_audio = st.session_state.get(f"orig_audio_{fid}")
            if orig_audio:
                st.download_button(
                    "Download original audio (.mp3)",
                    data=orig_audio,
                    file_name=f"{base_name}_orig.mp3",
                    mime="audio/mpeg",
                    key=f"dl_orig_audio_{fid}",
                )

        with aud_col2:
            if enable_translation and translated:
                if st.button(f"Prepare dubbed {target_lang_label} audio (.mp3)", key=f"prep_dub_audio_btn_{fid}"):
                    with st.spinner("Synthesizing dubbed audio — timing is approximate..."):
                        try:
                            lang_code = LANGUAGES[target_lang_label]
                            st.session_state[f"dub_audio_{fid}_{lang_code}"] = synthesize_dubbed_audio(
                                fid, lang_code, json.dumps(translated)
                            )
                        except Exception as e:
                            st.error(f"Dubbing failed: {e}")
                lang_code = LANGUAGES[target_lang_label]
                dub_audio = st.session_state.get(f"dub_audio_{fid}_{lang_code}")
                if dub_audio:
                    st.download_button(
                        f"Download dubbed {target_lang_label} audio (.mp3)",
                        data=dub_audio,
                        file_name=f"{base_name}_dubbed_{lang_code}.mp3",
                        mime="audio/mpeg",
                        key=f"dl_dub_audio_{fid}_{lang_code}",
                    )
                    st.caption(
                        "Auto-generated via edge-tts, with lines sped up (up to "
                        f"{MAX_RATE_SPEEDUP_PCT}%) when needed to fit their original time slot. "
                        "Still approximate — very fast/dense speech can still overlap."
                    )
            else:
                st.caption("Enable translation in the sidebar to also get a dubbed audio track.")

        st.divider()
        st.subheader("Export with burned-in captions")

        rendered_key = f"rendered_{fid}"
        lang_code = LANGUAGES[target_lang_label]
        suffix_tag = caption_suffix(enable_translation, translated, display_mode, lang_code)

        if is_video:
            st.caption("Hardcodes the captions currently shown above directly onto your uploaded_file video.")
            if st.button("Render captioned MP4", key=f"render_captioned_btn_{fid}"):
                srt_text = build_srt(display_segments)
                with st.spinner("Burning in captions..."):
                    try:
                        st.session_state[rendered_key] = render_video_with_captions(fid, suffix, srt_text)
                        st.session_state[f"{rendered_key}_name"] = f"{base_name}_captioned{suffix_tag}.mp4"
                    except Exception as e:
                        st.error(f"Rendering failed: {e}")
        else:
            st.caption("Synthesizes a simple captioned 'lyric video' from your audio.")
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                resolution_label = st.selectbox("Resolution", ["1280x720", "1920x1080", "854x480"], index=0, key=f"resolution_{fid}")
            with r_col2:
                bg_color = st.color_picker("Background color", "#000000", key=f"bgcolor_{fid}")
            with r_col3:
                waveform = st.checkbox("Add waveform", value=True, key=f"waveform_{fid}")
            if st.button("Render lyric video (MP4)", key=f"render_lyric_btn_{fid}"):
                srt_text = build_srt(display_segments)
                w, h = (int(x) for x in resolution_label.split("x"))
                with st.spinner("Rendering video — this can take a bit..."):
                    try:
                        st.session_state[rendered_key] = render_lyric_video(
                            fid, suffix, srt_text, w, h, bg_color, waveform
                        )
                        st.session_state[f"{rendered_key}_name"] = f"{base_name}_lyricvideo{suffix_tag}.mp4"
                    except Exception as e:
                        st.error(f"Rendering failed: {e}")

        rendered = st.session_state.get(rendered_key)
        if rendered:
            st.video(rendered)
            st.download_button(
                "Download captioned MP4",
                data=rendered,
                file_name=st.session_state.get(f"{rendered_key}_name", f"{base_name}_captioned.mp4"),
                mime="video/mp4",
                key=f"dl_rendered_video_{fid}",
            )


def run():
    st.title("🎙️ Upload Media + Synced Transcript")

    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox(
            "Whisper model size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=1,
            help="Larger models are more accurate but slower to transcribe.",
        )
        st.caption("Requires ffmpeg on PATH, plus `faster-whisper` and `deep-translator`.")

        st.header("Translation")
        enable_translation = st.checkbox("Translate captions", value=False)
        target_lang_label = st.selectbox(
            "Target language", list(LANGUAGES.keys()), index=0, disabled=not enable_translation,
        )
        display_mode = st.radio(
            "Caption display", ["Translated only", "Original only", "Both"],
            index=0, disabled=not enable_translation,
        )

    uploaded_files = st.file_uploader(
        "Upload audio or video files",
        type=SUPPORTED_EXTS,
        help="Supported: " + ", ".join(sorted(SUPPORTED_EXTS)),
        accept_multiple_files=True,
    )

    if uploaded_files:
        seen_hashes = {}
        deduped_files = []
        duplicates = []  # (duplicate_name, original_name)
        for f in uploaded_files:
            fid = file_hash(f.getvalue())
            if fid in seen_hashes:
                duplicates.append((f.name, seen_hashes[fid]))
                continue
            seen_hashes[fid] = f.name
            deduped_files.append(f)

        if duplicates:
            lines = "; ".join(f"'{dup}' is identical to '{orig}'" for dup, orig in duplicates)
            st.warning(
                f"Skipped {len(duplicates)} duplicate upload(s) with content already uploaded: {lines}. "
                "Only one copy of each is processed below."
            )
        uploaded_files = deduped_files

    if uploaded_files:
        if st.button(f"Transcribe All ({len(uploaded_files)} files)", key="transcribe_all_btn"):
            progress = st.progress(0.0, text="Starting batch transcription...")
            results = []  # (filename, success, message)

            for i, f in enumerate(uploaded_files):
                data = f.getvalue()
                suffix = os.path.splitext(f.name)[1].lower()

                if suffix not in EXT_INFO:
                    results.append((f.name, False, f"unsupported file type {suffix}"))
                    progress.progress((i + 1) / len(uploaded_files), text=f"Skipped {f.name} (unsupported type)")
                    continue

                fid = file_hash(data)
                ensure_saved(fid, data, suffix)
                progress.progress(
                    i / len(uploaded_files),
                    text=f"Transcribing {f.name} ({i + 1}/{len(uploaded_files)})...",
                )
                try:
                    transcript = transcribe(fid, suffix, model_size)
                    st.session_state[f"transcript_{fid}_{model_size}"] = transcript
                    results.append((f.name, True, None))
                except Exception as e:
                    results.append((f.name, False, str(e)))
                progress.progress((i + 1) / len(uploaded_files))

            progress.empty()
            successes = [r for r in results if r[1]]
            failures = [r for r in results if not r[1]]
            if successes:
                st.success(
                    f"Transcribed {len(successes)}/{len(results)} file(s) with the '{model_size}' model. "
                    "Open a tab to view, translate, or export it."
                )
            if failures:
                st.error("Failed: " + "; ".join(f"{name} ({msg})" for name, _, msg in failures))

        tab_labels = []
        seen = {}
        for f in uploaded_files:
            label = os.path.splitext(f.name)[0]
            if label in seen:
                seen[label] += 1
                label = f"{label} ({seen[label]})"
            else:
                seen[label] = 1
            tab_labels.append(label)

        tabs = st.tabs(tab_labels)
        for tab, uploaded_file in zip(tabs, uploaded_files):
            with tab:
                process_one_file(uploaded_file, model_size, enable_translation, target_lang_label, display_mode)
    else:
        st.info("Upload one or more audio/video files above to get started.")


if __name__ == "__main__":
    st.set_page_config(page_title="Media File Transcriber", layout="wide")
    run()
