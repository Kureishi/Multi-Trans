# Media Transcriber Suite

Transcribe, translate, caption, and dub video/audio — from a YouTube URL or from
your own uploaded files — all in Streamlit.

## What's in here

| File | What it does |
|---|---|
| `transcriber_app.py` | **Launcher.** Sidebar lets you pick "YouTube URL" or "Upload a File", then dispatches to one of the two apps below. Run this one. |
| `youtube_transcriber.py` | Paste a YouTube URL → embedded player with synced/overlaid captions, translation, dubbed audio, and MP4 export with burned-in captions. |
| `media_file_transcriber.py` | Upload your own audio/video file(s) → same feature set, plus **batch mode** (multiple files, each in its own tab). |
| `requirements.txt` | Python dependencies. |

Each app file is also fully runnable on its own (`streamlit run youtube_transcriber.py`), independent of the launcher.

## Features

- **Playback with synced captions**: transcript segments highlight and auto-scroll in time with the video/audio, and can also be overlaid directly on the video (or shown as a big "lyrics" line for audio-only files).
- **Transcription** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (choose model size: `tiny` → `large-v3`).
- **Translation** to 14 languages (including English) via Google Translate (free, through `deep-translator`), with a choice of showing Original / Translated / Both. Source language is auto-detected, so this also covers translating non-English audio *into* English.
- **Dubbed audio** via [edge-tts](https://github.com/rany2/edge-tts): synthesizes each translated line and times it to the original segment, speeding lines up (capped, to stay intelligible) when they'd otherwise run long and overlap the next one. This is an approximation of a real dub, not a polished one — see **Limitations** below.
- **Export**:
  - Downloadable `.txt` transcripts (original and/or translated).
  - Downloadable audio-only `.mp3` (original track, or the dubbed track).
  - Downloadable `.mp4` with captions burned in permanently (hardcoded via ffmpeg's `subtitles` filter). For audio-only uploads, this instead synthesizes a simple "lyric video" (solid background, optional waveform) so you still get a shareable video.
- **Batch mode** (file-upload app only): upload several files at once, each gets its own tab named after the file. A "Transcribe All" button queues transcription across all of them sequentially.
- **Broad format support** for uploads: audio `.mp3 .wav .m4a .aac .flac .ogg .opus .wma`, video `.mp4 .m4v .webm .mov .mkv .avi .flv`.
- **Sensible filenames**: downloads are named after the source (video title for YouTube, original filename for uploads), tagged with `_orig` / `_dubbed_<lang>` / `_transcript_<lang>` etc., so multiple downloads don't collide or get confusing.

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. ffmpeg (system dependency — not a pip package)

You need `ffmpeg` on your `PATH`, **built with libass support** (needed for the
`subtitles` filter that burns in captions). Check with:

```bash
ffmpeg -filters | grep subtitles      # macOS/Linux
ffmpeg -filters | findstr subtitles   # Windows
```

You should see lines for both `ass` and `subtitles` filters. If not, your ffmpeg
build is missing libass — this is common with minimal builds (e.g. some
conda-forge installs). Get a full build instead:

- **macOS**: `brew install ffmpeg` (Homebrew's build includes libass)
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg` (or build from a static source if your distro's package is minimal)
- **Windows**: grab a static build from the community builders linked on [ffmpeg.org's download page](https://ffmpeg.org/download.html) — e.g. [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) (`win64-gpl` build), extract it, and add its `bin` folder to your `PATH`.

If you have multiple ffmpeg installs (e.g. conda-forge *and* a full build),
Windows/macOS/Linux all just use whichever comes first in `PATH` — run
`ffmpeg -version` in a fresh terminal to confirm which one is active.

## Running it

```bash
streamlit run transcriber_app.py
```

This opens the launcher with a sidebar toggle between the YouTube and file-upload flows. Or run either app directly if you only need one:

```bash
streamlit run youtube_transcriber.py
streamlit run media_file_transcriber.py
```

## Usage notes

- **Transcribe first**, then optionally enable translation in the sidebar — translation re-runs automatically when you change the target language or display mode, but transcription itself only re-runs when you click the button (and only if the model size changed).
- **Batch mode**: upload multiple files, hit **"Transcribe All"** to queue transcription across all of them (sequential — Streamlit is single-threaded per session, so this isn't parallel), then open individual tabs to translate/export each one. Uploading the exact same file content twice is detected and skipped with a warning, rather than erroring.
- **MP4 export** re-encodes the video track (audio is stream-copied), so it's slower than transcription — expect it to take a while on longer/higher-resolution videos.
- Files embedded for the custom player/overlay in the upload app are inlined as base64, so very large uploads (100+ MB) will be slow to preview in-browser; transcription/translation/export aren't affected.

## Limitations

- **Translation** uses the free, unofficial Google Translate backend (via `deep-translator`) — good for everyday use, not a professional MT service, and requires internet access.
- **Source language is auto-detected**, not user-specified — this works for any spoken language (transcription and translation aren't locked to English), but manually specifying it when you already know it (e.g. `language="es"`) would improve both speed and accuracy. Not yet exposed as an option.
- **Dubbed audio** is an approximation: TTS pacing rarely matches the original speaker exactly. Lines are sped up by up to 50% to try to fit their original time slot; anything that would need more than that to fit will still overlap somewhat. Treat it as "get the gist in another language," not a lip-synced dub.
- **edge-tts** is an unofficial, reverse-engineered wrapper around Microsoft's Read Aloud service (same category of tool as the free Google Translate backend above) — free and keyless, but not an official/supported API.
- A few uploaded formats (`.wma`, `.mov`, `.mkv`, `.avi`, `.flv`) commonly don't play back natively in the browser's preview player even though transcription/translation/export all still work fine on them — the app flags this when it applies.
- `st.cache_data`/`st.cache_resource` are used throughout to avoid redoing expensive work (transcription, translation, rendering), keyed by content hash (uploads) or video ID (YouTube) plus the relevant settings — so switching a model size or target language will trigger fresh work the first time, then reuse it afterward.
