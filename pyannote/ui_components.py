import streamlit as st
import pandas as pd
from config import CUSTOM_CSS, SUPPORTED_WHISPER_MODELS, SUPPORTED_AUDIO_TYPES
from utils import check_cuda_availability, save_results, format_timestamp
import logging

logger = logging.getLogger(__name__)

def initialize_page():
    """Initialize the Streamlit page with basic configuration"""
    st.set_page_config(
        page_title="Audio Transcription & Speaker Diarization",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def setup_sidebar():
    """Setup and return sidebar configuration"""
    with st.sidebar:
        st.header("Configuration")
        
        # Hugging Face Token Input
        hf_token = st.text_input("Enter HuggingFace Token", type="password")
        if not hf_token:
            st.warning("Please enter your HuggingFace token")
            return None
        
        st.markdown("---")
        
        # Audio Input Selection
        input_type = st.radio(
            "Select Input Type",
            ["Upload Audio File", "Real-time Transcription"]
        )
        
        st.markdown("---")
        
        # GPU Configuration
        st.subheader("Hardware Configuration")
        cuda_available, cuda_info = check_cuda_availability()
        
        if cuda_available:
            st.success(f"✅ GPU Available: {cuda_info}")
            use_gpu = st.checkbox("Use GPU", value=True)
        else:
            st.warning("❌ GPU/CUDA not available. Using CPU only.")
            use_gpu = False
        
        st.markdown("---")
        
        # Model Configurations
        st.subheader("Whisper Configuration")
        whisper_model_size = st.selectbox(
            "Model Size",
            SUPPORTED_WHISPER_MODELS,
            help="Larger models are more accurate but slower"
        )
        
        whisper_language = st.text_input(
            "Language (optional, leave blank for auto-detection)",
            "",
            help="Enter language code (e.g., 'en' for English)"
        )
        
        whisper_task = st.selectbox(
            "Task",
            ["transcribe", "translate"],
            help="'translate' will translate to English"
        )
        
        st.markdown("---")
        
        # Diarization Configuration
        st.subheader("Diarization Configuration")
        use_num_speakers = st.checkbox(
            "Specify Number of Speakers",
            help="Enable to set exact number of speakers"
        )
        
        if use_num_speakers:
            num_speakers = st.number_input("Number of Speakers", min_value=1, value=2)
            min_speakers = None
            max_speakers = None
        else:
            num_speakers = None
            min_speakers = st.number_input("Minimum Speakers", min_value=1, value=1)
            max_speakers = st.number_input("Maximum Speakers", min_value=1, value=5)
        
        # Debug Options
        with st.expander("Advanced Options"):
            show_raw_output = st.checkbox(
                "Show Raw Output",
                help="Display detailed processing information"
            )
            show_debug_logs = st.checkbox(
                "Show Debug Logs",
                help="Display technical processing logs"
            )

        config = {
            'hf_token': hf_token,
            'input_type': input_type,
            'use_gpu': use_gpu,
            'whisper_model_size': whisper_model_size,
            'whisper_language': whisper_language if whisper_language else None,
            'whisper_task': whisper_task,
            'num_speakers': num_speakers,
            'min_speakers': min_speakers,
            'max_speakers': max_speakers,
            'show_raw_output': show_raw_output,
            'show_debug_logs': show_debug_logs,
            'sample_rate': 16000  # Add hardcoded sample rate
        }
        
        return config

def display_results(segments, container, show_debug=False):
    """Display transcription results in the specified container"""
    if not segments:
        container.warning("No transcription results produced.")
        return
    
    if show_debug:
        container.json(segments)
    
    # Display results
    current_speaker = None
    for segment in sorted(segments, key=lambda x: x['start']):
        if current_speaker != segment['speaker']:
            container.markdown(f"### {segment['speaker']}")
            current_speaker = segment['speaker']
        
        container.markdown(
            f"""
            <div class="timestamp">{segment['start_time']} - {segment['end_time']}</div>
            <div class="transcription-text">{segment['text']}</div>
            """,
            unsafe_allow_html=True
        )

def show_download_options(segments, container):
    """Display download options for the results"""
    if not segments:
        return
    
    container.markdown("---")
    container.subheader("📥 Download Results")
    
    col1, col2 = container.columns(2)
    
    # CSV download
    csv_data = save_results(segments, 'csv')
    if csv_data:
        with col1:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="transcription.csv",
                mime="text/csv"
            )
    
    # Text download
    text_data = save_results(segments, 'text')
    if text_data:
        with col2:
            st.download_button(
                label="Download Text",
                data=text_data,
                file_name="transcription.txt",
                mime="text/plain"
            )

def show_debug_info(diarization, transcription, container):
    """Display debug information"""
    with container.expander("Raw Diarization Segments"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            container.write(f"{speaker}: {turn.start:.2f} - {turn.end:.2f}")
    
    with container.expander("Raw Transcription"):
        for segment in transcription['segments']:
            container.write(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}")

def show_recording_controls():
    """Display recording controls and return button states"""
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("🎙️ Start Recording")
    with col2:
        stop_button = st.button("⏹️ Stop Recording")
    
    status = st.empty()
    results_container = st.container()
    
    return start_button, stop_button, status, results_container

def show_file_uploader():
    """Display file uploader and return uploaded file"""
    return st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        help="Supported formats: WAV, MP3"
    )