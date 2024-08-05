import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
from lightning_whisper_mlx import LightningWhisperMLX
import requests
from typing import List, Dict, Any
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
import subprocess
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit page config
st.set_page_config(page_title="Auto Subtitled Video Generator", page_icon=":movie_camera:", layout="wide")

# Define constants
TASK_VERBS = {
    "Transcribe": "Transcribing",
    "Translate": "Translating"
}

DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODELS = {
    "Tiny": "tiny",
    "Small": "small",
    "Distil Small (English)": "distil-small.en",
    "Base": "base",
    "Medium": "medium",
    "Distil Medium (English)": "distil-medium.en",
    "Large": "large",
    "Large v2": "large-v2",
    "Distil Large v2": "distil-large-v2",
    "Large v3": "large-v3",
    "Distil Large v3": "distil-large-v3"
}
APP_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_DIR = APP_DIR / "local_video"
LOCAL_DIR.mkdir(exist_ok=True)
SAVE_DIR = LOCAL_DIR / "output"
SAVE_DIR.mkdir(exist_ok=True)

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
        "ffmpeg",
        "-i", audio_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    return mx.array(audio_array)

def process_audio(model: LightningWhisperMLX, audio_path: str, task: str) -> Dict[str, Any]:
    logging.info(f"Processing audio with model: {model.name}, task: {task}")
    
    try:
        if task.lower() == "transcribe":
            results = model.transcribe(audio_path)
        elif task.lower() == "translate":
            results = model.transcribe(audio_path)  # LightningWhisperMLX doesn't have a separate translate method
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        logging.info(f"{task.capitalize()} completed successfully")
        return {
            "text": " ".join([segment[2] for segment in results["segments"]]),
            "segments": results["segments"]
        }
    except Exception as e:
        logging.error(f"Unexpected error in LightningWhisperMLX.transcribe: {e}")
        raise

def write_subtitles(segments: List[Dict[str, Any]], format: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "vtt":
            f.write("WEBVTT\n\n")
            for segment in segments:
                start = format_timestamp(segment[0])
                end = format_timestamp(segment[1])
                text = segment[2]
                f.write(f"{start} --> {end}\n")
                f.write(f"{text.strip()}\n\n")
        elif format == "srt":
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment[0], ms_separator=',')
                end = format_timestamp(segment[1], ms_separator=',')
                text = segment[2]
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text.strip()}\n\n")

def format_timestamp(seconds: float, ms_separator: str = '.') -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ms_separator)

def create_download_link(file_path: str, link_text: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

def main():
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
            - Optional English translation
            - Lightning-fast processing

            ### Choose your task
            - 🎙️ Transcribe: Capture spoken words in the original language
            - 🌍 Translate: Convert speech to English subtitles
        """)
    
    input_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    task = st.selectbox("Select Task", list(TASK_VERBS.keys()), index=0)
    
    # Update model selection
    selected_model = st.selectbox("Select Whisper Model", list(MODELS.keys()), index=0)
    MODEL_NAME = MODELS[selected_model]
    
    # Add quantization option
    quant = st.selectbox("Select Quantization", [None, "4bit", "8bit"], index=0)
    
    # Add batch size slider
    batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=12, step=1)
    
    if input_file and st.button(task):
        with st.spinner(f"{TASK_VERBS[task]} the video using {selected_model} model..."):
            try:
                # Save uploaded file
                input_path = str(SAVE_DIR / "input.mp4")
                with open(input_path, "wb") as f:
                    f.write(input_file.read())
                
                # Initialize LightningWhisperMLX
                model = LightningWhisperMLX(model=MODEL_NAME, batch_size=batch_size, quant=quant)
                
                # Process audio
                results = process_audio(model, input_path, task.lower())
                
                # Display results
                col3, col4 = st.columns(2)
                with col3:
                    st.video(input_file)
                
                # Write subtitles
                vtt_path = str(SAVE_DIR / "transcript.vtt")
                srt_path = str(SAVE_DIR / "transcript.srt")
                write_subtitles(results["segments"], "vtt", vtt_path)
                write_subtitles(results["segments"], "srt", srt_path)
                
                with col4:
                    st.text_area("Transcription", results["text"], height=300)
                    st.success(f"{task} completed successfully using {selected_model} model!")
                
                # Create zip file with outputs
                zip_path = str(SAVE_DIR / "transcripts.zip")
                with ZipFile(zip_path, "w") as zipf:
                    for file in [vtt_path, srt_path]:
                        zipf.write(file, os.path.basename(file))
                
                # Create download link
                st.markdown(create_download_link(zip_path, "Download Transcripts"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.exception("Error in main processing loop")

if __name__ == "__main__":
    main()