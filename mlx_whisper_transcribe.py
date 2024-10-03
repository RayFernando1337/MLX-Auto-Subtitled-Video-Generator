import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
import mlx_whisper
import requests
from typing import List, Dict, Any
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
import subprocess
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODELS = {
    "Tiny (Q4)": "mlx-community/whisper-tiny-mlx-q4",
    "Large v3": "mlx-community/whisper-large-v3-mlx",
    "Small English (Q4)": "mlx-community/whisper-small.en-mlx-q4",
    "Small (FP32)": "mlx-community/whisper-small-mlx-fp32",
    "Distil Large v3 (English)": "mlx-community/distil-whisper-large-v3",
    "Large v3 Turbo": "mlx-community/whisper-large-v3-turbo"
}
APP_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_DIR = APP_DIR / "local_video"
LOCAL_DIR.mkdir(exist_ok=True)
SAVE_DIR = LOCAL_DIR / "output"
SAVE_DIR.mkdir(exist_ok=True)
LANGUAGES = {
    "Detect automatically": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

# Utility functions
@st.cache_data
def load_lottie_url(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logging.error(f"Failed to load Lottie animation: {e}")
        return None

def prepare_audio(audio_path: str) -> mx.array:
    command = [
        "ffmpeg", "-i", audio_path, "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return mx.array(audio_array)

def process_audio(model_path: str, audio: mx.array, task: str, language: str = None) -> Dict[str, Any]:
    logging.info(f"Processing audio with model: {model_path}, task: {task}, language: {language}")
    try:
        decode_options = {"language": language} if language else {}

        if task.lower() == "transcribe":
            results = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model_path, fp16=False, verbose=True, word_timestamps=True, **decode_options
            )
            logging.info(f"{task.capitalize()} completed successfully")
            return results
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as e:
        logging.error(f"Unexpected error in mlx_whisper.{task}: {e}")
        raise

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if vtt:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    else:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace('.', ',')

def create_download_link(file_path: str, link_text: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Subtitle and transcription functions
def split_text_into_lines(text: str, max_chars: int = 42) -> List[str]:
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
        
        subtitle_count = 1
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            text = ' '.join(word['word'] for word in words)
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            
            lines = split_text_into_lines(text)
            
            for i in range(0, len(lines), 2):
                subtitle_lines = lines[i:i+2]
                subtitle_text = '\n'.join(subtitle_lines)
                
                start_index = sum(len(line.split()) for line in lines[:i])
                end_index = min(sum(len(line.split()) for line in lines[:i+2]), len(words))
                
                start_word = words[start_index]
                end_word = words[end_index - 1]
                
                start_time = start_word['start']
                end_time = end_word['end']
                
                duration = end_time - start_time
                min_duration = max(len(subtitle_text) / 21, 1.5)
                if duration < min_duration:
                    end_time = start_time + min_duration
                
                if format == "srt":
                    f.write(f"{subtitle_count}\n")
                    f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                elif format == "vtt":
                    f.write(f"{format_timestamp(start_time, vtt=True)} --> {format_timestamp(end_time, vtt=True)}\n")
                    f.write(f"{subtitle_text}\n\n")
                
                subtitle_count += 1
            
            check_data_loss(segment, lines)

    check_final_output(segments, output_file)

def write_text_transcription(segments: List[Dict[str, Any]], output_file: str, remove_fillers: bool = True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment['text']
            if remove_fillers:
                text = re.sub(r'\b(um|uh)\b', '', text).strip()
            f.write(text + "\n")

def check_data_loss(segment: Dict[str, Any], processed_lines: List[str]) -> None:
    processed_words = ' '.join(processed_lines).split()
    original_words = ' '.join(word['word'] for word in segment['words']).split()
    if len(processed_words) != len(original_words):
        logging.warning(f"Potential data loss detected in segment {segment.get('id', 'unknown')}")
        logging.warning(f"Original: {' '.join(original_words)}")
        logging.warning(f"Processed: {' '.join(processed_words)}")

def check_final_output(segments: List[Dict[str, Any]], output_file: str) -> None:
    original_text = ' '.join(seg['text'] for seg in segments)
    final_text = ' '.join(line.strip() for line in open(output_file, 'r', encoding='utf-8').readlines() if line.strip() and not line[0].isdigit() and '-->' not in line)
    if original_text != final_text:
        logging.warning("Potential data loss or word order change detected in final output")

# Streamlit UI functions
def render_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        lottie = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_HjK9Ol.json")
        if lottie:
            st_lottie(lottie)
    with col2:
        st.markdown("""
            ## Apple MLX Powered Video Transcription

            Upload your video and get:
            - Accurate transcripts (SRT/VTT files)
            - Lightning-fast processing

            🎙️ Transcribe: Capture spoken words in the original language
        """)

def render_model_selection():
    selected_model = st.selectbox("Select Whisper Model", list(MODELS.keys()), index=5)
    if selected_model == "Distil Large v3 (English)":
        st.info("""
        **Distil Large v3 Model**
        
        This new model offers significant performance improvements:
        - Runs approximately 40 times faster than real-time on M1 Max chips
        - Can transcribe 12 minutes of audio in just 18 seconds
        - Provides a great balance between speed and accuracy
        
        Ideal for processing longer videos or when you need quick results without sacrificing too much accuracy.
        """)
    if selected_model == "Large v3 Turbo":
        st.info("""
        **Large v3 Turbo**
        
        This new model offers significant performance improvements:
        - Transcribes 12 minutes in 14 seconds on an M2 Ultra (~50X faster than real time)
        - Significantly smaller than the Large v3 model (809M vs 1550M)
        - It is multilingual
        """)
    if selected_model in ["Small English (Q4)", "Distil Large v3 (English)"]:
        return MODELS[selected_model], True
    else:
        return MODELS[selected_model], False

def process_video(input_file, model_name, language):
    try:
        input_path = str(SAVE_DIR / "input.mp4")
        with open(input_path, "wb") as f:
            f.write(input_file.read())
        
        audio = prepare_audio(input_path)
        results = process_audio(model_name, audio, "transcribe", language)
        
        vtt_path = str(SAVE_DIR / "transcript.vtt")
        srt_path = str(SAVE_DIR / "transcript.srt")
        txt_path = str(SAVE_DIR / "transcript.txt")
        
        write_subtitles(results["segments"], "vtt", vtt_path)
        write_subtitles(results["segments"], "srt", srt_path)
        write_text_transcription(results["segments"], txt_path)
        
        zip_path = str(SAVE_DIR / "transcripts.zip")
        with ZipFile(zip_path, "w") as zipf:
            for file in [vtt_path, srt_path, txt_path]:
                zipf.write(file, os.path.basename(file))
        
        st.markdown(create_download_link(zip_path, "Download Transcripts"), unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.video(input_file)
        with col4:
            st.text_area("Transcription", results["text"], height=300)
        
        st.success(f"Transcription completed successfully using {model_name} model!")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("Error in processing video")

def main():
    st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")
    render_header()
    input_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    model_name, is_language_locked = render_model_selection()
   
    if is_language_locked:
        selected_language = "English"
        st.selectbox("Select language", ["English"], disabled=True)
    else:
        selected_language = st.selectbox("Select language", list(LANGUAGES.keys()))
    language = LANGUAGES[selected_language]

    if input_file and st.button("Transcribe"):
        with st.spinner(f"Transcribing the video using {model_name} model..."):
            process_video(input_file, model_name, language)

if __name__ == "__main__":
    main()