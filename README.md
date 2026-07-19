# Media Transcriber Suite

Transcribe, translate, caption, and dub video/audio — from a URL (YouTube and
hundreds of other sites, via yt-dlp) or from your own uploaded files — all in
Streamlit.

## What's in here

| File | What it does |
|---|---|
| `transcriber_app.py` | **Launcher.** Sidebar lets you pick "Video/audio URL" or "Upload a File", then dispatches to one of the two apps below. Run this one. |
| `youtube_transcriber.py` | Paste video/audio URL(s) — YouTube or hundreds of other sites via yt-dlp (one or many, one per line) → player(s) with synced/overlaid captions, translation, dubbed audio, and MP4 export with burned-in captions. Batch mode: one tab per source. |
| `media_file_transcriber.py` | Upload your own audio/video file(s) → same feature set, plus **batch mode** (multiple files, each in its own tab). |
| `cli.py` | Command-line interface — same engine as both apps above, no browser needed. Good for scripting/automation. |
| `mtt_ui.py` | Console-script wrapper — installed as `mtt-ui`, launches the Streamlit UI (`transcriber_app.py`) as a real pip-installed command. |
| `pyproject.toml` | Packaging config — installs `cli.py` and `mtt_ui.py` as standalone `mtt` / `mtt-ui` commands (see below). |
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
- **Sensible filenames**: downloads are named after the source (title for URLs, original filename for uploads), tagged with `_orig` / `_dubbed_<lang>` / `_transcript_<lang>` etc., so multiple downloads don't collide or get confusing.

## CLI

`cli.py` is a thin wrapper around the exact same functions the two Streamlit
apps use — it imports them directly, so there's no separate implementation to
drift out of sync. Must live in the same folder as `youtube_transcriber.py`
and `media_file_transcriber.py`.

```bash
# Transcribe two local files to plain text
python cli.py --input_type file --source clip1.mp3 clip2.mp4 --output_type txt

# Transcribe + translate a YouTube (or any yt-dlp-supported) video to Japanese, get a captioned MP4 and a dubbed MP3
python cli.py --input_type url --source https://youtu.be/XXXXXXXXXXX \
    --output_type mp4 mp3 --target-lang Japanese

# Batch: several files, every output type, translated to English
python cli.py --input_type file --source a.mp3 b.mp4 c.wav \
    --output_type txt mp3 mp4 --target-lang English --model small

# See available target languages
python cli.py --list-languages
```

Required flags: `--input_type {file,url}`, `--source` (one or more paths/URLs), `--output_type` (one or more of `txt mp3 mp4`).

Useful optional flags: `--model` (Whisper size, default `base`), `--target-lang` (enables translation), `--display-mode` (`"Original only"` / `"Translated only"` / `"Both"` — controls what goes into `.txt`/`.mp4` output when translating), `--quality` (URL-source MP4 export resolution), `--resolution`/`--bg-color`/`--waveform` (lyric-video export for audio file uploads), `--output-dir` (default `./output`).

Behaves like the UI's batch mode: duplicate sources are detected and skipped (by a stable source ID for URLs — derived from actual site metadata, not just the literal URL string, so different URL shapes for the same video are still caught — or by absolute path for files), URL downloads get a short pause between them, and a failed source is logged and skipped rather than aborting the whole run — exit code is `1` if anything failed, `0` otherwise.

### Installing it as standalone `mtt` / `mtt-ui` commands

`pyproject.toml` registers two console-script entry points:
- `cli.py`'s `main()` as **`mtt`** — the CLI, as described above.
- `mtt_ui.py`'s `main()` as **`mtt-ui`** — launches the Streamlit UI (`transcriber_app.py`) the same way `streamlit run transcriber_app.py` would, just callable as a plain installed command from anywhere. Any extra arguments (e.g. `mtt-ui --server.port 8502`) are forwarded straight through to Streamlit.

An editable install gives you both, with no need to be inside the project folder or remember file paths:

```bash
pip install -e .
mtt --list-languages                 # CLI, works from anywhere
mtt-ui                               # opens the Streamlit UI in your browser
```

This is also the scaffold for turning this into a proper published pip package later — `pyproject.toml` already declares the dependencies and both entry points; publishing would mainly mean choosing a real package name/version and running through `build`/`twine` (or similar), rather than restructuring anything here.

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

This opens the launcher with a sidebar toggle between the URL and file-upload flows. Or run either app directly if you only need one:

```bash
streamlit run youtube_transcriber.py
streamlit run media_file_transcriber.py
```

## Usage notes

- **Transcribe first**, then optionally enable translation in the sidebar — translation re-runs automatically when you change the target language or display mode, but transcription itself only re-runs when you click the button (and only if the model size changed).
- **Batch mode**: both apps support processing multiple items at once — paste several video/audio URLs (one per line) or upload multiple files — each gets its own tab named after its source. **"Transcribe All"** queues transcription across all of them sequentially (Streamlit is single-threaded per session, so this isn't parallel), then open individual tabs to translate/export each one. Duplicate items (same source, or identical file content) are detected and skipped with a warning rather than erroring. Video export (MP4 with burned-in captions) stays a deliberate per-tab action in both apps — it's the heaviest operation, so it's not batched.
- **MP4 export** re-encodes the video track (audio is stream-copied), so it's slower than transcription — expect it to take a while on longer/higher-resolution videos.
- Files embedded for the custom player/overlay in the upload app are inlined as base64, so very large uploads (100+ MB) will be slow to preview in-browser; transcription/translation/export aren't affected.
- URL batch mode is network-bound (each source is actually downloaded), so it's noticeably slower per item than file batch mode and scales with source count accordingly. There's a short delay between downloads to be gentler on the hosting site, but a large batch can still take a while — and is more exposed to site-side rate limiting than local files ever would be.

## Limitations

- **Non-YouTube URL support comes from yt-dlp**, which covers roughly 1,800 sites (Vimeo, Twitter/X, TikTok, SoundCloud, Twitch VODs, direct `.mp4`/`.mp3` links, etc. — see [yt-dlp's supported-sites list](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)). A few things worth knowing:
  - **Only download content you have the rights to use.** Platform terms of service vary a lot on this, and checking that is on the person using the tool — this app doesn't and can't enforce it.
  - **Site support can break.** It depends on yt-dlp's actively maintained per-site "extractors," which stop working when a platform changes its internals until yt-dlp is updated. "Works with any URL" means "any URL yt-dlp currently supports," not a permanent guarantee — a failure may be a temporarily broken extractor rather than a bug here.
  - **Private/login-gated content isn't supported.** yt-dlp can do this with extra cookie/auth setup, but that's meaningfully more scope than what's implemented here.
  - **Preview differs by source.** YouTube gets the full experience — captions overlaid live on an embedded, scrubbable player. Everything else has no public embeddable player API to reuse, so there's no preview *before* transcribing; once transcribed (which already downloads the audio), you get a native audio player with a synced, lyrics-style caption line instead — same pattern as an uploaded audio file. MP4 export (burning captions onto the actual video) still works for any supported source, YouTube or not — it just isn't part of the live preview.
- **Translation** uses the free, unofficial Google Translate backend (via `deep-translator`) — good for everyday use, not a professional MT service, and requires internet access.
- **Source language is auto-detected**, not user-specified — this works for any spoken language (transcription and translation aren't locked to English), but manually specifying it when you already know it (e.g. `language="es"`) would improve both speed and accuracy. Not yet exposed as an option.
- **Dubbed audio** is an approximation: TTS pacing rarely matches the original speaker exactly. Lines are sped up by up to 50% to try to fit their original time slot; anything that would need more than that to fit will still overlap somewhat. Treat it as "get the gist in another language," not a lip-synced dub.
- **edge-tts** is an unofficial, reverse-engineered wrapper around Microsoft's Read Aloud service (same category of tool as the free Google Translate backend above) — free and keyless, but not an official/supported API.
- A few uploaded formats (`.wma`, `.mov`, `.mkv`, `.avi`, `.flv`) commonly don't play back natively in the browser's preview player even though transcription/translation/export all still work fine on them — the app flags this when it applies.
- `st.cache_data`/`st.cache_resource` are used throughout to avoid redoing expensive work (transcription, translation, rendering), keyed by content hash (uploads) or the actual URL (URL sources) plus the relevant settings — so switching a model size or target language will trigger fresh work the first time, then reuse it afterward.
