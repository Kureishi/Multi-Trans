"""
Media Transcriber Suite — launcher
====================================

A thin front door that lets the user choose which transcription flow to use:

    1. YouTube URL           -> youtube_transcriber.run()
    2. Upload a file (mp3/mp4) -> media_file_transcriber.run()

Both underlying apps remain fully independent and can still be run directly
via `streamlit run youtube_transcriber.py` or
`streamlit run media_file_transcriber.py` — this file just imports their
`run()` functions and dispatches to whichever one the user picks, so all
three files must live in the same folder:

    transcriber_app.py
    youtube_transcriber.py
    media_file_transcriber.py

Run with:
    streamlit run transcriber_app.py

Requirements (pip install):
    streamlit
    faster-whisper
    yt-dlp
    deep-translator

System requirement:
    ffmpeg on PATH, built with libass support (needed for the `subtitles`
    filter used by both apps' "burn in captions" export feature).
"""

import streamlit as st

st.set_page_config(page_title="Media Transcriber Suite", layout="wide")

st.sidebar.title("🎬 Media Transcriber Suite")
source = st.sidebar.radio(
    "Choose input source",
    ["YouTube URL", "Upload a File"],
)
st.sidebar.divider()

if source == "YouTube URL":
    import youtube_transcriber
    youtube_transcriber.run()
else:
    import media_file_transcriber
    media_file_transcriber.run()
