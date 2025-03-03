import streamlit as st
import torch
import torchaudio
import whisper  # Corrected import
from pyannote.audio import Pipeline
import numpy as np
import tempfile
import os
from datetime import datetime

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.fromtimestamp(seconds).strftime('%H:%M:%S'))

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

# Set page configuration
st.set_page_config(
    page_title="Audio Transcription & Diarization",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS to improve appearance
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
    </style>
    """, unsafe_allow_html=True)

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_audio(audio_path, pipeline, whisper_model, num_speakers=None, min_speakers=None, 
                 max_speakers=None, whisper_language=None, whisper_task="transcribe"):
    """Process audio file with diarization and transcription"""
    
    # First, get the full transcription
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    audio = audio.squeeze().numpy()
    
    # Get full transcription
    transcription = whisper_model.transcribe(
        audio,
        language=whisper_language if whisper_language else None,
        task=whisper_task
    )
    
    # Get diarization
    diarization = pipeline(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    return diarization, transcription

def merge_transcription_with_diarization(diarization, transcription):
    """Merge transcription with speaker diarization"""
    combined_results = []
    
    # Convert diarization to list of segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Find matching transcription segments
        matching_segments = []
        for segment in transcription["segments"]:
            # Check if segments overlap
            if (segment["start"] <= turn.end and segment["end"] >= turn.start):
                matching_segments.append(segment["text"])
        
        if matching_segments:
            combined_results.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "text": " ".join(matching_segments)
            })
    
    return combined_results

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

    # Main content area
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if audio_file is not None:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
            
        try:
            # Loading models
            progress_bar.progress(20)
            with st.spinner("Loading models..."):
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                
                if torch.cuda.is_available():
                    pipeline = pipeline.to(torch.device("cuda"))
                
                whisper_model = whisper.load_model(whisper_model_size)
            
            # Processing audio
            progress_bar.progress(40)
            with st.spinner("Processing audio..."):
                diarization, transcription = process_audio(
                    audio_path,
                    pipeline,
                    whisper_model,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    whisper_language=whisper_language if whisper_language else None,
                    whisper_task=whisper_task
                )
                
                # Merge results
                progress_bar.progress(70)
                combined_results = merge_transcription_with_diarization(diarization, transcription)
            
            # Display results
            progress_bar.progress(90)
            st.markdown("## 📝 Transcription Results")
            
            # Convert to DataFrame
            df = pd.DataFrame(combined_results)
            
            # Format timestamps
            df['start_time'] = df['start'].apply(lambda x: format_timestamp(x))
            df['end_time'] = df['end'].apply(lambda x: format_timestamp(x))
            
            # Display results in a more organized way
            current_speaker = None
            for _, row in df.iterrows():
                if current_speaker != row['speaker']:
                    st.markdown(f"### {row['speaker']}")
                    current_speaker = row['speaker']
                
                st.markdown(
                    f"""
                    <div class="timestamp">{row['start_time']} - {row['end_time']}</div>
                    <div>{row['text']}</div>
                    <br>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Download options
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            # CSV download
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
            for _, row in df.iterrows():
                if current_speaker != row['speaker']:
                    text_content += f"\n\n{row['speaker']}:\n"
                    current_speaker = row['speaker']
                text_content += f"[{row['start_time']} - {row['end_time']}] {row['text']}\n"
            
            st.download_button(
                label="Download as Text",
                data=text_content,
                file_name="transcription.txt",
                mime="text/plain"
            )
            
            progress_bar.progress(100)
            
        finally:
            # Cleanup
            os.unlink(audio_path)
            
        st.success("Processing completed successfully!")

if __name__ == "__main__":
    main()