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
from pydub import AudioSegment
import io
import time

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

def chunk_audio(audio_file, chunk_duration_ms=10000):
    """Split audio into chunks"""
    try:
        # Read the uploaded file content
        audio_bytes = audio_file.read()
        
        # Create a temporary file for the original audio
        file_extension = audio_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_original:
            tmp_original.write(audio_bytes)
            original_path = tmp_original.name

        # Load the audio using pydub
        try:
            # Try loading based on file extension
            audio = AudioSegment.from_file(original_path, format=file_extension)
        except:
            # If that fails, try automatic format detection
            audio = AudioSegment.from_file(original_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Set frame rate to 16000 Hz if different
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        chunks = []
        
        # Split into chunks
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)

        return chunks

    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        logger.error(f"Error in chunk_audio: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'original_path' in locals():
            try:
                os.unlink(original_path)
            except:
                pass

def process_chunk(chunk, pipeline, whisper_model, use_gpu=False):
    """Process a single audio chunk"""
    try:
        # Create temporary file for the chunk
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_chunk:
            # Export with specific parameters
            chunk.export(
                tmp_chunk.name,
                format='wav',
                parameters=[
                    "-ac", "1",  # mono
                    "-ar", "16000"  # 16kHz sample rate
                ]
            )
            chunk_path = tmp_chunk.name

        # Load audio for Whisper
        audio = whisper.load_audio(chunk_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

        # Detect language
        _, probs = whisper_model.detect_language(mel)
        detected_language = max(probs, key=probs.__getitem__)

        # Decode audio
        options = whisper.DecodingOptions(
            fp16=use_gpu,
            language=detected_language
        )
        
        result = whisper.decode(whisper_model, mel, options)

        # Create transcription dict
        transcription = {
            'text': result.text,
            'segments': [{
                'text': result.text,
                'start': 0,
                'end': len(audio) / whisper.audio.SAMPLE_RATE
            }]
        }

        # Perform diarization
        diarization = pipeline(chunk_path)

        return diarization, transcription

    except Exception as e:
        logger.error(f"Error in process_chunk: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup
        if 'chunk_path' in locals():
            try:
                os.unlink(chunk_path)
            except:
                pass

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
            # Use the most overlapping speaker for this transcription segment
            max_overlap = 0
            best_speaker = None
            best_segment = None
            
            for diar_seg in overlapping_diar:
                overlap_start = max(trans_start, diar_seg['start'])
                overlap_end = min(trans_end, diar_seg['end'])
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = diar_seg['speaker']
                    best_segment = diar_seg

            if best_segment:
                segments.append({
                    'speaker': best_speaker,
                    'start': trans_start,
                    'end': trans_end,
                    'text': trans_text
                })
    
    return segments

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

    # Main content area
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    if audio_file is not None:
        # Debug information
        st.write("File details:", {
            "name": audio_file.name,
            "type": audio_file.type,
            "size": audio_file.size
        })

        try:
            # Display audio file
            st.audio(audio_file)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            transcription_container = st.empty()

            # Load models
            status_text.text("Loading models...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            if use_gpu and torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
            
            whisper_model = whisper.load_model(whisper_model_size)
            if use_gpu and torch.cuda.is_available():
                whisper_model = whisper_model.to(torch.device("cuda"))

            # Reset file pointer
            audio_file.seek(0)

            # Split audio into chunks
            status_text.text("Splitting audio into chunks...")
            chunks = chunk_audio(audio_file, chunk_duration_ms=10000)  # 10-second chunks
            total_chunks = len(chunks)
            
            all_segments = []
            current_display = ""
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                
                try:
                    # Process chunk
                    diarization, transcription = process_chunk(
                        chunk,
                        pipeline,
                        whisper_model,
                        use_gpu=use_gpu
                    )
                    
                    # Get segments for this chunk
                    chunk_segments = get_segments_from_diarization(diarization, transcription)
                    
                    # Adjust timestamps for chunks
                    chunk_offset = i * 10  # 10 seconds per chunk
                    for segment in chunk_segments:
                        segment['start'] += chunk_offset
                        segment['end'] += chunk_offset
                        segment['start_time'] = format_timestamp(segment['start'])
                        segment['end_time'] = format_timestamp(segment['end'])
                    
                    all_segments.extend(chunk_segments)
                    
                    # Update display
                    current_display = ""
                    current_speaker = None
                    for segment in all_segments:
                        if current_speaker != segment['speaker']:
                            current_display += f"\n### {segment['speaker']}\n"
                            current_speaker = segment['speaker']
                        
                        current_display += f"**{segment['start_time']} - {segment['end_time']}**\n{segment['text']}\n\n"
                    
                    transcription_container.markdown(current_display)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    st.warning(f"Warning: Error processing chunk {i+1}. Continuing with next chunk...")
                    continue
                
                # Add a small delay to simulate real-time processing
                time.sleep(0.1)
            
            # Final display and download options
            status_text.text("Processing completed!")
            
            # Download options
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            # CSV download
            df = pd.DataFrame(all_segments)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="transcription.csv",
                mime="text/csv"
            )
            
            # Text format download
            text_content = current_display
            st.download_button(
                label="Download as Text",
                data=text_content,
                file_name="transcription.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error("Processing error", exc_info=True)

if __name__ == "__main__":
    main()