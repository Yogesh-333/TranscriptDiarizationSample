import streamlit as st
import logging
import tempfile
import os
from ui_components import (
    initialize_page, setup_sidebar, display_results, 
    show_download_options, show_debug_info, 
    show_recording_controls, show_file_uploader
)
from utils import (
    initialize_models, process_audio_file, format_timestamp,
    save_results
)
from audio_handlers import RealTimeTranscriber, AudioRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_file_upload(audio_file, pipeline, whisper_model, config):
    """Handle uploaded audio file processing"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'audio_file.wav')
    
    try:
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            f.write(audio_file.getvalue())
        
        st.info("Processing audio file... This may take a moment.")
        
        # Process the audio
        segments, diarization, transcription = process_audio_file(
            temp_path,
            pipeline,
            whisper_model,
            config
        )
        
        # Show debug information
        if config['show_raw_output'] or not segments:
            st.subheader("Debug Information")
            
            with st.expander("Raw Diarization Segments", expanded=True):
                count = 0
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    st.write(f"{speaker}: {format_timestamp(turn.start)} - {format_timestamp(turn.end)}")
                    count += 1
                st.info(f"Total diarization segments: {count}")
            
            if transcription:
                with st.expander("Raw Transcription", expanded=True):
                    for segment in transcription['segments']:
                        st.write(f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}: {segment['text']}")
                    st.info(f"Total transcription segments: {len(transcription['segments'])}")
        
        # Display results
        if segments:
            st.markdown("## 📝 Transcription Results")
            display_results(segments, st, config['show_debug_logs'])
            show_download_options(segments, st)
        else:
            st.warning("""
            No segments were produced from the audio. This might be because:
            1. The audio quality is too low
            2. The speakers are not distinct enough
            3. The audio file format is not properly supported
            
            Try:
            1. Using a different audio file
            2. Converting the file to WAV format
            3. Ensuring the audio is clear and speakers are distinct
            4. Checking the debug information above
            """)
        
        return segments
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None
        
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

def handle_realtime_recording(pipeline, whisper_model, config):
    """Handle real-time recording and transcription"""
    try:
        # Initialize transcriber and recorder
        transcriber = RealTimeTranscriber(whisper_model, pipeline)
        recorder = AudioRecorder(transcriber)
        
        # Show recording controls
        start_button, stop_button, status, results_container = show_recording_controls()
        
        # Initialize or get session state
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        
        # Handle recording start
        if start_button:
            status.markdown('<p class="recording-active">🔴 Recording...</p>', unsafe_allow_html=True)
            if recorder.start():
                st.session_state.recording = True
                logger.info("Recording started successfully")
            else:
                status.error("Failed to start recording")
        
        # Handle recording stop
        if stop_button and st.session_state.recording:
            logger.info("Stopping recording")
            recorder.stop()
            status.info("Processing recording...")
            
            # Process the recording
            segments = transcriber.process_audio(config)
            
            if segments:
                with results_container:
                    st.markdown("## 📝 Transcription Results")
                    display_results(segments, st, config['show_debug_logs'])
                    show_download_options(segments, st)
            else:
                status.warning("No transcription results were produced. Try speaking louder or adjusting your microphone.")
            
            st.session_state.recording = False
            status.empty()
            
    except Exception as e:
        logger.error(f"Error in real-time recording: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def main():
    try:
        # Initialize page
        initialize_page()
        st.title("🎤 Audio Transcription & Speaker Diarization")
        st.markdown("---")
        
        # Setup sidebar and get configuration
        config = setup_sidebar()
        if not config:
            return
        
        # Initialize models
        try:
            with st.spinner("Loading models..."):
                pipeline, whisper_model = initialize_models(
                    config['hf_token'],
                    config['whisper_model_size'],
                    config['use_gpu']
                )
                logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            st.error(f"Error loading models: {str(e)}")
            return
        
        # Handle different input types
        if config['input_type'] == "Upload Audio File":
            st.subheader("Upload Audio")
            audio_file = show_file_uploader()
            
            if audio_file is not None:
                handle_file_upload(audio_file, pipeline, whisper_model, config)
        
        else:  # Real-time Transcription
            st.subheader("Real-time Audio Transcription")
            st.info("""
            🎙️ Click 'Start Recording' to begin.
            Speak clearly into your microphone.
            Click 'Stop Recording' when finished to see the results.
            """)
            
            handle_realtime_recording(pipeline, whisper_model, config)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()