import streamlit as st
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
import numpy as np
import tempfile
import os
from datetime import datetime
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Audio Transcription & Diarization",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .speaker-header {
        color: #1f77b4;
        margin-top: 20px;
    }
    .timestamp {
        color: #666;
        font-size: 0.9em;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .debug-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .transcription-text {
        font-size: 1.1em;
        margin: 5px 0;
        padding: 5px;
        border-left: 3px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

def check_cuda_availability():
    """Check CUDA and GPU availability"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        return True, f"CUDA {cuda_version} with {gpu_name}"
    return False, "CUDA not available"

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_audio(audio_path, pipeline, whisper_model, num_speakers=None, min_speakers=None, 
                 max_speakers=None, whisper_language=None, whisper_task="transcribe", use_gpu=False):
    """Process audio file with diarization and transcription"""
    
    logger.info("Starting audio processing")
    
    # Load and resample audio
    audio, sr = torchaudio.load(audio_path)
    logger.info(f"Loaded audio with sample rate: {sr}")
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
        logger.info("Resampled audio to 16kHz")
    
    audio = audio.squeeze().numpy()
    
    # Device configuration
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Transcribe with whisper
    logger.info("Starting Whisper transcription")
    transcription = whisper_model.transcribe(
        audio,
        language=whisper_language if whisper_language else None,
        task=whisper_task,
        fp16=use_gpu
    )
    logger.info("Completed Whisper transcription")
    
    # Perform diarization
    logger.info("Starting diarization")
    diarization = pipeline(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    logger.info("Completed diarization")
    
    return diarization, transcription

def merge_diarization_segments(diarization):
    """Merge consecutive segments from the same speaker in diarization"""
    merged_segments = []
    current_segment = None
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if not current_segment:
            current_segment = {
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            }
        elif current_segment['speaker'] == speaker and \
             (turn.start - current_segment['end']) < 0.1:  # Merge if gap is less than 0.1 seconds
            current_segment['end'] = turn.end
        else:
            merged_segments.append(current_segment)
            current_segment = {
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            }
    
    if current_segment:
        merged_segments.append(current_segment)
    
    return merged_segments

def get_segments_from_diarization(diarization, transcription):
    """Extract segments from diarization and transcription"""
    # First, merge diarization segments
    merged_diar_segments = merge_diarization_segments(diarization)
    
    # Debug print
    logger.info("Merged Diarization Segments:")
    for seg in merged_diar_segments:
        logger.info(f"{seg['speaker']}: {format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}")
    
    segments = []
    
    # Process each transcription segment
    for trans_segment in transcription['segments']:
        trans_text = trans_segment['text'].strip()
        trans_start = trans_segment['start']
        trans_end = trans_segment['end']
        
        # Find all merged diarization segments that overlap with this transcription
        overlapping_diar = []
        for diar_seg in merged_diar_segments:
            if (diar_seg['start'] <= trans_end and diar_seg['end'] >= trans_start):
                overlapping_diar.append(diar_seg)
        
        if overlapping_diar:
            # Split the transcription text among the overlapping segments
            for i, diar_seg in enumerate(overlapping_diar):
                overlap_start = max(trans_start, diar_seg['start'])
                overlap_end = min(trans_end, diar_seg['end'])
                
                # Calculate the portion of text that belongs to this segment
                if len(overlapping_diar) == 1:
                    text_portion = trans_text
                else:
                    # Split text based on time overlap
                    overlap_duration = overlap_end - overlap_start
                    total_duration = sum(min(d['end'], trans_end) - max(d['start'], trans_start) 
                                      for d in overlapping_diar)
                    ratio = overlap_duration / total_duration
                    words = trans_text.split()
                    word_count = max(1, int(len(words) * ratio))
                    text_portion = " ".join(words[:word_count])
                    trans_text = " ".join(words[word_count:])
                
                segments.append({
                    'speaker': diar_seg['speaker'],
                    'start_time': format_timestamp(diar_seg['start']),
                    'end_time': format_timestamp(diar_seg['end']),
                    'start': diar_seg['start'],
                    'end': diar_seg['end'],
                    'text': text_portion.strip()
                })
    
    # Sort segments by start time
    segments.sort(key=lambda x: x['start'])
    
    # Merge segments with the same time boundaries
    final_segments = []
    for segment in segments:
        if not final_segments or \
           final_segments[-1]['speaker'] != segment['speaker'] or \
           final_segments[-1]['end'] != segment['start']:
            final_segments.append(segment)
        else:
            final_segments[-1]['text'] += ' ' + segment['text']
            final_segments[-1]['end'] = segment['end']
            final_segments[-1]['end_time'] = segment['end_time']
    
    return final_segments

def main():
    st.title("🎤 Audio Transcription & Speaker Diarization")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Hugging Face Token Input
        hf_token = st.text_input("Enter HuggingFace Token", type="password")
        if not hf_token:
            st.warning("Please enter your HuggingFace token")
            st.stop()

        st.markdown("---")
        
        # GPU Configuration
        st.subheader("Hardware Configuration")
        cuda_available, cuda_info = check_cuda_availability()
        
        if cuda_available:
            st.success(f"✅ GPU Available: {cuda_info}")
            use_gpu = st.checkbox("Use GPU", value=True)
        else:
            st.warning("❌ GPU/CUDA not available. Using CPU only.")
            st.info("""To enable GPU support, install CUDA and torch with CUDA support:
            1. Install NVIDIA CUDA Toolkit
            2. Install PyTorch with CUDA: 
               `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
            """)
            use_gpu = False

        st.markdown("---")
        
        # Model Configurations
        st.subheader("Whisper Configuration")
        whisper_model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large"]
        )
        
        whisper_language = st.text_input(
            "Language (optional, leave blank for auto-detection)",
            ""
        )
        
        whisper_task = st.selectbox(
            "Task",
            ["transcribe", "translate"]
        )

        st.markdown("---")
        
        # Diarization Configuration
        st.subheader("Diarization Configuration")
        use_num_speakers = st.checkbox("Specify Number of Speakers")
        
        if use_num_speakers:
            num_speakers = st.number_input("Number of Speakers", min_value=1, value=2)
            min_speakers = None
            max_speakers = None
        else:
            num_speakers = None
            min_speakers = st.number_input("Minimum Speakers", min_value=1, value=1)
            max_speakers = st.number_input("Maximum Speakers", min_value=1, value=5)
        
        # Debug Options
        with st.expander("Debug Options"):
            show_raw_output = st.checkbox("Show Raw Output")
            show_debug_logs = st.checkbox("Show Debug Logs")

    # Main content area
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if audio_file is not None:
        # Create a progress bar and status containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_log = st.empty() if show_debug_logs else None
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
            
        try:
            # Loading models
            status_text.text("Loading models...")
            progress_bar.progress(20)
            
            if debug_log:
                debug_log.info("Initializing pipeline and models...")
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            if use_gpu and torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
            
            whisper_model = whisper.load_model(whisper_model_size)
            if use_gpu and torch.cuda.is_available():
                whisper_model = whisper_model.to(torch.device("cuda"))
            
            if debug_log:
                debug_log.info("Models loaded successfully")
            
            # Processing audio
            status_text.text("Processing audio...")
            progress_bar.progress(40)
            
            diarization, transcription = process_audio(
                audio_path,
                pipeline,
                whisper_model,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                whisper_language=whisper_language if whisper_language else None,
                whisper_task=whisper_task,
                use_gpu=use_gpu
            )
            
            # Debug information
            if show_raw_output:
                st.subheader("Debug Information")
                
                with st.expander("Raw Diarization Segments"):
                    diarization_count = 0
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        st.write(f"{speaker}: {format_timestamp(turn.start)} - {format_timestamp(turn.end)}")
                        diarization_count += 1
                    st.info(f"Total diarization segments: {diarization_count}")
                
                with st.expander("Raw Transcription Segments"):
                    for i, segment in enumerate(transcription['segments']):
                        st.write(f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}: {segment['text']}")
                    st.info(f"Total transcription segments: {len(transcription['segments'])}")
            
            # Get segments
            status_text.text("Processing results...")
            progress_bar.progress(70)
            segments = get_segments_from_diarization(diarization, transcription)
            
            # Display results
            status_text.text("Preparing display...")
            progress_bar.progress(90)
            
            st.markdown("## 📝 Transcription Results")
            
            if not segments:
                st.warning("No segments were produced. Try adjusting the configuration:")
                st.markdown("""
                - Specify the exact number of speakers if known
                - Try a larger Whisper model
                - Check if the audio is clear and speakers are distinct
                - Try specifying the language
                """)
                
                if debug_log:
                    debug_log.error("No segments were produced from the processing")
            else:
                # Display results
                current_speaker = None
                for segment in segments:
                    if current_speaker != segment['speaker']:
                        st.markdown(f"### {segment['speaker']}")
                        current_speaker = segment['speaker']
                    
                    st.markdown(
                        f"""
                        <div class="timestamp">{segment['start_time']} - {segment['end_time']}</div>
                        <div class="transcription-text">{segment['text']}</div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Download options
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                # CSV download
                df = pd.DataFrame(segments)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="transcription.csv",
                    mime="text/csv"
                )
                
                # Text format download
                text_content = ""
                current_speaker = None
                for segment in segments:
                    if current_speaker != segment['speaker']:
                        text_content += f"\n\n{segment['speaker']}:\n"
                        current_speaker = segment['speaker']
                    text_content += f"[{segment['start_time']} - {segment['end_time']}] {segment['text']}\n"
                
                st.download_button(
                    label="Download as Text",
                    data=text_content,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
            
            progress_bar.progress(100)
            status_text.text("Processing completed!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if debug_log:
                debug_log.exception("Error during processing")
            logger.error("Processing error", exc_info=True)
            
        finally:
            # Cleanup
            try:
                os.unlink(audio_path)
            except:
                pass

if __name__ == "__main__":
    main()