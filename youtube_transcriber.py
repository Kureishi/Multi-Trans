"""
YouTube Video Player + Synced Transcript (faster-whisper)
==========================================================

Run with:
    streamlit run youtube_transcriber.py

Requirements (pip install):
    streamlit
    faster-whisper
    yt-dlp
    deep-translator
    edge-tts

System requirement:
    ffmpeg must be installed and on PATH (used by yt-dlp to extract audio,
    and by faster-whisper/PyAV to decode it).
    - macOS:   brew install ffmpeg
    - Ubuntu:  sudo apt-get install ffmpeg
    - Windows: https://ffmpeg.org/download.html

How it works:
    1. You paste a YouTube URL. It's embedded via the YouTube IFrame Player API.
    2. Clicking "Transcribe audio" downloads just the audio track with yt-dlp,
       then runs faster-whisper to get timestamped segments.
    3. A small JS component polls the player's current time and highlights /
       auto-scrolls to the matching transcript line in real time, and shows
       it as a subtitle-style overlay burned onto the video.
    4. Optionally, translate the transcript to another language (via Google
       Translate, using the free `deep-translator` package) and overlay the
       translated captions instead of / alongside the original.
    5. Optionally, render and download an actual .mp4 file with the current
       captions burned into the video permanently (downloads the full video
       with yt-dlp, then uses ffmpeg's `subtitles` filter to hardcode them).
       This requires an ffmpeg build with libass support (the common default
       builds from ffmpeg.org / most package managers include it).
"""

import os
import re
import json
import tempfile
import subprocess

import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str):
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    from faster_whisper import WhisperModel
    # CPU / int8 for broad compatibility. If you have a GPU, change to
    # device="cuda", compute_type="float16" for much faster transcription.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@st.cache_data(show_spinner=False)
def download_audio(video_id: str) -> str:
    import yt_dlp

    out_dir = tempfile.mkdtemp()
    out_template = os.path.join(out_dir, f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            return os.path.join(out_dir, f)
    raise FileNotFoundError("Audio extraction failed — check that ffmpeg is installed.")


@st.cache_data(show_spinner=False)
def transcribe(video_id: str, model_size: str):
    audio_path = download_audio(video_id)
    model = load_model(model_size)
    segments, _info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]


def fmt_time(t: float) -> str:
    t = int(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
def translate_segments(video_id: str, model_size: str, target_lang_code: str):
    """Translate each transcript segment's text to the target language.

    Uses deep-translator's free Google Translate backend. Batches requests
    in chunks to stay well under request-size limits for long transcripts.
    """
    from deep_translator import GoogleTranslator

    segments = transcribe(video_id, model_size)
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


QUALITY_HEIGHTS = {"480p": 480, "720p": 720, "1080p": 1080, "Best available": None}


@st.cache_data(show_spinner=False)
def download_video(video_id: str, quality: str) -> str:
    """Download the full video (video+audio, muxed to mp4) at the requested quality."""
    import yt_dlp

    height = QUALITY_HEIGHTS.get(quality)
    out_dir = tempfile.mkdtemp()
    out_template = os.path.join(out_dir, f"{video_id}.%(ext)s")
    if height:
        fmt = (
            f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={height}][ext=mp4]/best"
        )
    else:
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for f in os.listdir(out_dir):
        if f.endswith(".mp4"):
            return os.path.join(out_dir, f)
    raise FileNotFoundError("Video download failed — check that ffmpeg is installed.")


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


def burn_subtitles(video_path: str, srt_text: str, out_path: str):
    work_dir = os.path.dirname(out_path)
    srt_path = os.path.join(work_dir, "captions.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    # ffmpeg's subtitles filter treats the path as a filter argument, so
    # colons and backslashes need escaping (matters especially on Windows).
    escaped_path = srt_path.replace("\\", "/").replace(":", "\\:")
    style = "FontName=Arial,FontSize=20,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1,Shadow=0,MarginV=30"
    vf = f"subtitles='{escaped_path}':force_style='{style}'"

    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-c:a", "copy", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


@st.cache_data(show_spinner=False)
def render_captioned_video(video_id: str, quality: str, srt_text: str) -> bytes:
    video_path = download_video(video_id, quality)
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"{video_id}_captioned.mp4")
    burn_subtitles(video_path, srt_text, out_path)
    with open(out_path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def extract_original_audio_mp3(video_id: str) -> bytes:
    """Just the original audio track, no video — reuses the already-downloaded wav."""
    wav_path = download_audio(video_id)
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"{video_id}.mp3")
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-q:a", "2", out_path]
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
def synthesize_dubbed_audio(video_id: str, lang_code: str, segments_json: str) -> bytes:
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
        # Time slot this line has to fit into before the next one starts.
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
                    pass  # fall back to the unsped-up version

        delay_ms = max(0, int(start * 1000))
        inputs.append(final_path)
        filter_parts.append(f"[{idx}:a]adelay={delay_ms}:all=1[a{idx}]")
        idx += 1

    if idx == 0:
        raise RuntimeError("No translated text available to synthesize.")

    mix_labels = "".join(f"[a{i}]" for i in range(idx))
    filter_complex = ";".join(filter_parts) + f";{mix_labels}amix=inputs={idx}:normalize=0[mixed]"

    out_path = os.path.join(work_dir, f"{video_id}_dubbed.mp3")
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", filter_complex, "-map", "[mixed]", "-c:a", "libmp3lame", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    with open(out_path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def get_video_title(video_id: str) -> str:
    """A filesystem-safe, human-readable name for this video, for use in
    download filenames — falls back to the video ID if lookup fails."""
    try:
        import yt_dlp
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title") or video_id
    except Exception:
        title = video_id
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")
    return safe[:60] or video_id


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

def run():
    st.title("🎬 YouTube Video + Synced Transcript")

    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox(
            "Whisper model size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=1,
            help="Larger models are more accurate but slower to transcribe.",
        )
        st.caption(
            "Requires ffmpeg on PATH, plus the `faster-whisper` and `yt-dlp` "
            "Python packages."
        )

        st.header("Translation")
        enable_translation = st.checkbox("Translate captions", value=False)
        target_lang_label = st.selectbox(
            "Target language",
            list(LANGUAGES.keys()),
            index=0,
            disabled=not enable_translation,
        )
        display_mode = st.radio(
            "Caption display",
            ["Translated only", "Original only", "Both"],
            index=0,
            disabled=not enable_translation,
        )
        st.caption("Translation uses Google Translate via the free `deep-translator` package (requires internet).")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Couldn't parse a YouTube video ID from that URL.")
            st.stop()

        if st.session_state.get("video_id") != video_id:
            st.session_state.video_id = video_id
            st.session_state.transcript = None

        base_name = get_video_title(video_id)

        col1, col2 = st.columns([1, 3])
        with col1:
            transcribe_clicked = st.button("Transcribe audio", type="primary")

        if transcribe_clicked:
            with st.spinner("Downloading audio and transcribing — this can take a while for long videos..."):
                try:
                    st.session_state.transcript = transcribe(video_id, model_size)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.session_state.transcript = None

        transcript = st.session_state.get("transcript")

        translated = None
        if transcript and enable_translation:
            target_lang_code = LANGUAGES[target_lang_label]
            with st.spinner(f"Translating to {target_lang_label}..."):
                try:
                    translated = translate_segments(video_id, model_size, target_lang_code)
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
            else:  # Original only
                display_segments = transcript
        else:
            display_segments = transcript

        show_overlay = st.checkbox("Show subtitle overlay on video", value=True)
        segments_json = json.dumps(display_segments or [])
        overlay_enabled_js = "true" if show_overlay else "false"

        html_code = f"""
        <div id="app">
          <div id="player-wrap" style="position:relative; width:100%; max-width:720px; aspect-ratio:16/9; background:#000; border-radius:8px; overflow:hidden;">
            <div id="player" style="width:100%; height:100%;"></div>
            <div id="caption-overlay" style="
                position:absolute; left:50%; bottom:6%; transform:translateX(-50%);
                max-width:88%; text-align:center; pointer-events:none;
                background: rgba(0,0,0,0.7); color:#fff; padding:6px 14px;
                border-radius:6px; font-family:sans-serif; font-size:1.05em;
                line-height:1.35; white-space:pre-line; z-index:5; display:none;"></div>
          </div>
          <div id="transcript-box" style="
              height: 320px; overflow-y: auto; margin-top: 12px;
              border: 1px solid #444; border-radius: 8px; padding: 10px;
              font-family: sans-serif; font-size: 14px; background: #111; color: #eee; white-space:pre-line;">
            <div id="transcript-inner"></div>
          </div>
        </div>

        <script>
          const segments = {segments_json};
          const overlayEnabled = {overlay_enabled_js};
          let player;
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
            if (player && player.seekTo) {{
              player.seekTo(t, true);
              player.playVideo();
            }}
          }}

          function renderTranscript() {{
            const inner = document.getElementById('transcript-inner');
            if (!segments.length) {{
              inner.innerHTML = "<em>No transcript yet. Click 'Transcribe audio' in the app.</em>";
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

          // Robust init: don't rely solely on window.onYouTubeIframeAPIReady, since
          // that callback can race (or silently never fire) inside an embedded
          // component iframe, especially if the script is cached. Poll instead.
          function createPlayer() {{
            if (player) return;
            player = new YT.Player('player', {{
              height: '100%',
              width: '100%',
              videoId: '{video_id}',
              playerVars: {{ playsinline: 1 }},
              events: {{ 'onReady': onPlayerReady }}
            }});
          }}

          function waitForYT() {{
            if (window.YT && window.YT.Player) {{
              createPlayer();
            }} else {{
              setTimeout(waitForYT, 100);
            }}
          }}

          // Keep the official callback too — harmless if it fires (createPlayer
          // just no-ops if already created), and covers the normal-case timing.
          window.onYouTubeIframeAPIReady = createPlayer;

          (function loadApi() {{
            const tag = document.createElement('script');
            tag.src = "https://www.youtube.com/iframe_api";
            document.head.appendChild(tag);
          }})();

          waitForYT();

          function onPlayerReady(event) {{
            setInterval(updateHighlight, 400);
          }}

          function updateHighlight() {{
            if (!player || !player.getCurrentTime || !segments.length) return;
            const t = player.getCurrentTime();
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
              if (idx >= 0) {{
                const el = document.getElementById('seg-' + idx);
                if (el) {{
                  el.style.background = '#2b3a55';
                  el.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
                }}
                overlay.textContent = segments[idx].text;
                overlay.style.display = overlayEnabled ? 'block' : 'none';
              }} else {{
                overlay.style.display = 'none';
              }}
              currentIdx = idx;
            }}
          }}

          renderTranscript();
        </script>
        """

        components.html(html_code, height=800, scrolling=False)

        if transcript:
            full_text = "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in transcript)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Download original transcript (.txt)",
                    data=full_text,
                    file_name=f"{base_name}_transcript_orig.txt",
                    mime="text/plain",
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
                    )

            st.divider()
            st.subheader("🔊 Download audio only")
            st.caption("No video, just an audio file — more than the plain-text transcript, less than a full video.")

            aud_col1, aud_col2 = st.columns(2)

            with aud_col1:
                if st.button("Prepare original audio (.mp3)"):
                    with st.spinner("Extracting original audio..."):
                        try:
                            st.session_state[f"orig_audio_{video_id}"] = extract_original_audio_mp3(video_id)
                        except Exception as e:
                            st.error(f"Audio extraction failed: {e}")
                orig_audio = st.session_state.get(f"orig_audio_{video_id}")
                if orig_audio:
                    st.download_button(
                        "Download original audio (.mp3)",
                        data=orig_audio,
                        file_name=f"{base_name}_orig.mp3",
                        mime="audio/mpeg",
                    )

            with aud_col2:
                if enable_translation and translated:
                    if st.button(f"Prepare dubbed {target_lang_label} audio (.mp3)"):
                        with st.spinner("Synthesizing dubbed audio — timing is approximate..."):
                            try:
                                lang_code = LANGUAGES[target_lang_label]
                                st.session_state[f"dub_audio_{video_id}_{lang_code}"] = synthesize_dubbed_audio(
                                    video_id, lang_code, json.dumps(translated)
                                )
                            except Exception as e:
                                st.error(f"Dubbing failed: {e}")
                    lang_code = LANGUAGES[target_lang_label]
                    dub_audio = st.session_state.get(f"dub_audio_{video_id}_{lang_code}")
                    if dub_audio:
                        st.download_button(
                            f"Download dubbed {target_lang_label} audio (.mp3)",
                            data=dub_audio,
                            file_name=f"{base_name}_dubbed_{lang_code}.mp3",
                            mime="audio/mpeg",
                        )
                        st.caption(
                            "Auto-generated via edge-tts, with lines sped up (up to "
                            f"{MAX_RATE_SPEEDUP_PCT}%) when needed to fit their original time slot. "
                            "Still approximate — very fast/dense speech can still overlap."
                        )
                else:
                    st.caption("Enable translation in the sidebar to also get a dubbed audio track.")

            st.divider()
            st.subheader("Export video with burned-in captions")
            st.caption(
                "Downloads the full video and hardcodes the captions currently shown "
                "above (original / translated / both, per your sidebar settings) into "
                "a new .mp4. This downloads much more data than transcription alone "
                "and can take a while for long videos."
            )
            quality = st.selectbox("Video quality", list(QUALITY_HEIGHTS.keys()), index=1)
            render_clicked = st.button("Render MP4 with captions")

            if render_clicked:
                srt_text = build_srt(display_segments)
                lang_code = LANGUAGES[target_lang_label]
                suffix = caption_suffix(enable_translation, translated, display_mode, lang_code)
                with st.spinner("Downloading video and burning in captions — this can take several minutes..."):
                    try:
                        video_bytes = render_captioned_video(video_id, quality, srt_text)
                        st.session_state.rendered_video = video_bytes
                        st.session_state.rendered_video_name = f"{base_name}_captioned{suffix}.mp4"
                    except Exception as e:
                        st.error(f"Rendering failed: {e}")

            if st.session_state.get("rendered_video"):
                st.video(st.session_state.rendered_video)
                st.download_button(
                    "Download captioned MP4",
                    data=st.session_state.rendered_video,
                    file_name=st.session_state.get("rendered_video_name", f"{base_name}_captioned.mp4"),
                    mime="video/mp4",
                )
    else:
        st.info("Paste a YouTube URL above to get started.")


if __name__ == "__main__":
    st.set_page_config(page_title="YouTube Transcriber", layout="wide")
    run()
