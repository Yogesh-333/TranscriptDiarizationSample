import streamlit as st
import numpy as np
import sounddevice as sd
import threading
import queue
import wave
import tempfile
import os
from datetime import datetime
import whisper
from scipy import signal
from sklearn.cluster import KMeans
import librosa

class AudioTranscriber:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.model = None
        self.chunk_duration = 5  # seconds
        
    def initialize_model(self, model_name="base"):
        try:
            self.model = whisper.load_model(model_name)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def detect_speakers(self, audio_data):
        """Simple speaker detection using energy-based segmentation"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=self.sample_rate, n_mfcc=13)
            
            # Perform clustering on MFCCs
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(mfccs.T)
            
            # Find the dominant speaker for this segment
            return "Speaker_" + str(np.argmax(np.bincount(clusters)))
        except Exception as e:
            print(f"Speaker detection error: {e}")
            return "Unknown Speaker"

    def process_audio(self):
        temp_dir = tempfile.mkdtemp()
        
        try:
            while self.is_recording:
                # Collect audio chunks
                audio_chunks = []
                start_time = datetime.now()
                
                while (datetime.now() - start_time).seconds < self.chunk_duration:
                    if not self.is_recording:
                        break
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        continue

                if audio_chunks:
                    # Process collected audio
                    audio_data = np.concatenate(audio_chunks, axis=0)
                    
                    # Save temporary WAV file
                    temp_file = os.path.join(temp_dir, f"temp_{datetime.now().timestamp()}.wav")
                    with wave.open(temp_file, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                    try:
                        # Transcribe audio
                        result = self.model.transcribe(temp_file)
                        
                        # Detect speaker
                        speaker = self.detect_speakers(audio_data)
                        
                        # Add transcription with timestamp and speaker
                        if result["text"].strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            transcription = f"[{timestamp}] {speaker}: {result['text']}"
                            st.session_state.transcriptions.append(transcription)
                    
                    except Exception as e:
                        st.error(f"Transcription error: {e}")
                    
                    finally:
                        # Cleanup temporary file
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)

        finally:
            # Cleanup temporary directory
            try:
                os.rmdir(temp_dir)
            except:
                pass

    def start_recording(self):
        self.is_recording = True
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

    def _record(self):
        with sd.InputStream(channels=self.channels,
                          samplerate=self.sample_rate,
                          callback=self.audio_callback):
            while self.is_recording:
                sd.sleep(100)

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()

def main():
    st.title("Real-time Speech Transcription with Speaker Detection")

    # Initialize session state
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = AudioTranscriber()
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    # Model selection
    model_size = st.sidebar.selectbox(
        "Select Whisper Model",
        ["tiny", "base", "small", "medium"],
        index=1
    )

    # Load model if not loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading Whisper model..."):
            if st.session_state.transcriber.initialize_model(model_size):
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")
                return

    # Recording controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🎤 ' + ('Stop Recording' if st.session_state.recording else 'Start Recording')):
            if not st.session_state.recording:
                st.session_state.recording = True
                st.session_state.transcriber.start_recording()
            else:
                st.session_state.recording = False
                st.session_state.transcriber.stop_recording()

    with col2:
        if st.button("🗑️ Clear Transcriptions"):
            st.session_state.transcriptions = []

    # Show recording status
    if st.session_state.recording:
        st.markdown("🔴 **Recording in progress...**")

    # Display transcriptions
    st.markdown("### Transcriptions:")
    for trans in reversed(st.session_state.transcriptions):
        st.markdown(f"- {trans}")

    # Export options
    if st.session_state.transcriptions:
        export_format = st.selectbox("Export Format", ["TXT", "JSON"])
        
        if st.button("Export"):
            if export_format == "TXT":
                content = "\n".join(st.session_state.transcriptions)
                st.download_button(
                    "Download TXT",
                    content,
                    "transcription.txt",
                    "text/plain"
                )
            else:
                import json
                content = json.dumps(st.session_state.transcriptions, indent=2)
                st.download_button(
                    "Download JSON",
                    content,
                    "transcription.json",
                    "application/json"
                )

    # Auto-refresh
    if st.session_state.recording:
        st.empty()
        time.sleep(0.1)
        st.experimental_rerun()

if __name__ == "__main__":
    main()