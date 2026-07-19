#!/usr/bin/env python3
"""
Media Transcriber Suite — CLI
===============================

Transcribe, translate, and export captions/dubs/burned-in-caption video from
the command line. This is a thin wrapper: it imports the actual functions
from youtube_transcriber.py and media_file_transcriber.py directly, so it's
always running the exact same transcription/translation/rendering engine as
the Streamlit apps — nothing here is a separate reimplementation.

Must live in the same folder as youtube_transcriber.py and
media_file_transcriber.py.

Examples
--------
Transcribe two local files to plain text:
    python cli.py --input_type file --source clip1.mp3 clip2.mp4 --output_type txt

Transcribe + translate a YouTube (or any yt-dlp-supported) video to Japanese, get a captioned MP4 and a dubbed MP3:
    python cli.py --input_type url --source https://youtu.be/XXXXXXXXXXX \\
        --output_type mp4 mp3 --target-lang Japanese

Batch: several files, every output type, translated to English:
    python cli.py --input_type file --source a.mp3 b.mp4 c.wav \\
        --output_type txt mp3 mp4 --target-lang English --model small

List available target languages:
    python cli.py --list-languages
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Streamlit's cache_data/cache_resource print a harmless "No runtime found"
# warning when used outside a running Streamlit app (which is exactly how
# this CLI uses them). The warning actually fires at *decoration time* (when
# youtube_transcriber.py/media_file_transcriber.py define their @st.cache_data
# functions during import), not at call time — so streamlit must be imported
# and its specific logger silenced *before* importing our two modules, or
# it's too late. Streamlit also sets an explicit level on these exact child
# loggers during its own setup, so silencing the generic "streamlit" logger
# instead doesn't stick — it has to target these precise names.
import streamlit as _st  # noqa: F401  (import triggers streamlit's logger setup)
for _logger_name in (
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.caching.cache_resource_api",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import youtube_transcriber as yt
import media_file_transcriber as mf

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
DISPLAY_MODE_CHOICES = ["Original only", "Translated only", "Both"]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def resolve_display_segments(transcript, translated, display_mode):
    if not translated:
        return transcript
    if display_mode == "Translated only":
        return translated
    if display_mode == "Both":
        return [
            {"start": o["start"], "end": o["end"], "text": f"{o['text']}\n{t['text']}"}
            for o, t in zip(transcript, translated)
        ]
    return transcript  # "Original only"


def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return path


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# File sources
# ---------------------------------------------------------------------------

def process_file_source(path, args):
    print(f"\n=== {path} ===")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "rb") as f:
        data = f.read()
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in mf.EXT_INFO:
        raise ValueError(f"Unsupported file type: {suffix}")

    _mime, is_video, _preview_ok = mf.EXT_INFO[suffix]
    fid = mf.file_hash(data)
    mf.ensure_saved(fid, data, suffix)
    base_name = sanitize(os.path.splitext(os.path.basename(path))[0]) or fid

    print(f"Transcribing ({args.model})...")
    transcript = mf.transcribe(fid, suffix, args.model)

    translated, lang_code = None, None
    if args.target_lang:
        lang_code = mf.LANGUAGES[args.target_lang]
        print(f"Translating to {args.target_lang}...")
        translated = mf.translate_segments(fid, suffix, args.model, lang_code)

    display_mode = args.display_mode or ("Translated only" if translated else "Original only")
    display_segments = resolve_display_segments(transcript, translated, display_mode)

    outputs = []

    if "txt" in args.output_type:
        orig_txt = "\n".join(f"[{mf.fmt_time(s['start'])}] {s['text']}" for s in transcript)
        outputs.append(write_text(
            os.path.join(args.output_dir, f"{base_name}_transcript_orig.txt"), orig_txt))
        if translated:
            tr_txt = "\n".join(f"[{mf.fmt_time(s['start'])}] {s['text']}" for s in translated)
            outputs.append(write_text(
                os.path.join(args.output_dir, f"{base_name}_transcript_{lang_code}.txt"), tr_txt))

    if "mp3" in args.output_type:
        print("Extracting original audio...")
        audio_bytes = mf.extract_original_audio_mp3(fid, suffix, is_video)
        outputs.append(write_bytes(
            os.path.join(args.output_dir, f"{base_name}_orig.mp3"), audio_bytes))
        if translated:
            print("Synthesizing dubbed audio (edge-tts, approximate timing)...")
            dub_bytes = mf.synthesize_dubbed_audio(fid, lang_code, json.dumps(translated))
            outputs.append(write_bytes(
                os.path.join(args.output_dir, f"{base_name}_dubbed_{lang_code}.mp3"), dub_bytes))

    if "mp4" in args.output_type:
        srt_text = mf.build_srt(display_segments)
        suffix_tag = mf.caption_suffix(bool(args.target_lang), translated, display_mode, lang_code or "")
        if is_video:
            print("Burning captions onto video...")
            video_bytes = mf.render_video_with_captions(fid, suffix, srt_text)
            outputs.append(write_bytes(
                os.path.join(args.output_dir, f"{base_name}_captioned{suffix_tag}.mp4"), video_bytes))
        else:
            print("Rendering lyric video...")
            w, h = (int(x) for x in args.resolution.split("x"))
            video_bytes = mf.render_lyric_video(fid, suffix, srt_text, w, h, args.bg_color, args.waveform)
            outputs.append(write_bytes(
                os.path.join(args.output_dir, f"{base_name}_lyricvideo{suffix_tag}.mp4"), video_bytes))

    return outputs


# ---------------------------------------------------------------------------
# URL sources (YouTube or any other yt-dlp-supported site)
# ---------------------------------------------------------------------------

def process_url_source(url, args):
    print(f"\n=== {url} ===")
    try:
        meta = yt.probe_url(url)
    except Exception as e:
        raise ValueError(f"Couldn't process this URL: {e}")
    if not meta["is_youtube"]:
        print(
            "Note: not a YouTube URL — using yt-dlp's general site support. "
            "Only download content you have the rights to use; platform terms of "
            "service vary, site support depends on yt-dlp's maintained extractors "
            "(which can break when a platform changes things), and private/"
            "login-gated content isn't supported."
        )
    base_name = sanitize(meta["title"])[:60] or yt.get_source_id(meta, url)

    print(f"Downloading audio and transcribing ({args.model})...")
    transcript = yt.transcribe(url, args.model)

    translated, lang_code = None, None
    if args.target_lang:
        lang_code = yt.LANGUAGES[args.target_lang]
        print(f"Translating to {args.target_lang}...")
        translated = yt.translate_segments(url, args.model, lang_code)

    display_mode = args.display_mode or ("Translated only" if translated else "Original only")
    display_segments = resolve_display_segments(transcript, translated, display_mode)

    outputs = []

    if "txt" in args.output_type:
        orig_txt = "\n".join(f"[{yt.fmt_time(s['start'])}] {s['text']}" for s in transcript)
        outputs.append(write_text(
            os.path.join(args.output_dir, f"{base_name}_transcript_orig.txt"), orig_txt))
        if translated:
            tr_txt = "\n".join(f"[{yt.fmt_time(s['start'])}] {s['text']}" for s in translated)
            outputs.append(write_text(
                os.path.join(args.output_dir, f"{base_name}_transcript_{lang_code}.txt"), tr_txt))

    if "mp3" in args.output_type:
        print("Extracting original audio...")
        audio_bytes = yt.extract_original_audio_mp3(url)
        outputs.append(write_bytes(
            os.path.join(args.output_dir, f"{base_name}_orig.mp3"), audio_bytes))
        if translated:
            print("Synthesizing dubbed audio (edge-tts, approximate timing)...")
            dub_bytes = yt.synthesize_dubbed_audio(url, lang_code, json.dumps(translated))
            outputs.append(write_bytes(
                os.path.join(args.output_dir, f"{base_name}_dubbed_{lang_code}.mp3"), dub_bytes))

    if "mp4" in args.output_type:
        print(f"Downloading full video ({args.quality}) and burning in captions — this can take a while...")
        srt_text = yt.build_srt(display_segments)
        suffix_tag = yt.caption_suffix(bool(args.target_lang), translated, display_mode, lang_code or "")
        video_bytes = yt.render_captioned_video(url, args.quality, srt_text)
        outputs.append(write_bytes(
            os.path.join(args.output_dir, f"{base_name}_captioned{suffix_tag}.mp4"), video_bytes))

    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Transcribe, translate, dub, and caption audio/video from files or video/audio URLs "
                    "(YouTube and hundreds of other sites supported by yt-dlp).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_type", choices=["file", "url"],
                        help="Whether --source entries are local file paths or video/audio URLs.")
    parser.add_argument("--source", nargs="+",
                        help="One or more file paths or video/audio URLs (space-separated).")
    parser.add_argument("--output_type", nargs="+", choices=["txt", "mp3", "mp4"],
                        help="One or more output types (space-separated).")
    parser.add_argument("--model", default="base", choices=MODEL_CHOICES,
                        help="faster-whisper model size (default: base).")
    parser.add_argument("--target-lang", default=None, choices=list(mf.LANGUAGES.keys()),
                        help="Translate to this language. Omit to keep the original language only.")
    parser.add_argument("--display-mode", default=None, choices=DISPLAY_MODE_CHOICES,
                        help="Which captions to use for .txt/.mp4 output when translating "
                             "(default: 'Translated only' if --target-lang is set, else 'Original only').")
    parser.add_argument("--quality", default="720p", choices=list(yt.QUALITY_HEIGHTS.keys()),
                        help="Video quality for URL-source MP4 export (default: 720p).")
    parser.add_argument("--resolution", default="1280x720",
                        help="Resolution (WxH) for lyric-video export from audio file uploads (default: 1280x720).")
    parser.add_argument("--bg-color", default="#000000",
                        help="Background color for lyric-video export (default: #000000).")
    parser.add_argument("--waveform", action="store_true",
                        help="Add a waveform visualization to lyric-video export.")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory to write output files to (default: ./output).")
    parser.add_argument("--list-languages", action="store_true",
                        help="Print available --target-lang options and exit.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_languages:
        for name in mf.LANGUAGES:
            print(name)
        return

    if not args.input_type or not args.source or not args.output_type:
        parser.error("--input_type, --source, and --output_type are required "
                     "(unless using --list-languages).")

    os.makedirs(args.output_dir, exist_ok=True)

    # De-duplicate sources up front, same as the batch-mode UI does.
    seen, sources = {}, []
    for src in args.source:
        if args.input_type == "url":
            try:
                meta = yt.probe_url(src)
                dedup_key = yt.get_source_id(meta, src)
            except Exception:
                dedup_key = src  # let it fail properly later with a clear per-source error
        else:
            dedup_key = os.path.abspath(src)
        if dedup_key in seen:
            print(f"Skipping duplicate source: {src} (same as {seen[dedup_key]})", file=sys.stderr)
            continue
        seen[dedup_key] = src
        sources.append(src)

    all_outputs = []
    failures = []

    for i, src in enumerate(sources):
        try:
            if args.input_type == "file":
                outputs = process_file_source(src, args)
            else:
                outputs = process_url_source(src, args)
                if i < len(sources) - 1:
                    time.sleep(1.5)  # be a little gentler on hosting sites between downloads
            all_outputs.extend(outputs)
            for p in outputs:
                print(f"  -> {p}")
        except Exception as e:
            print(f"[FAILED] {src}: {e}", file=sys.stderr)
            failures.append((src, str(e)))

    print(f"\nDone. {len(all_outputs)} file(s) written to {os.path.abspath(args.output_dir)}.")
    if failures:
        print(f"{len(failures)} source(s) failed:", file=sys.stderr)
        for src, msg in failures:
            print(f"  - {src}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
