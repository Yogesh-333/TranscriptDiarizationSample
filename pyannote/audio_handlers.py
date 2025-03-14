import pyaudio
import numpy as np
import threading
import queue
import tempfile
import os
from scipy.io import wavfile
import logging
from config import SAMPLE_RATE, CHUNK_SIZE, FORMAT, CHANNELS
from utils import process_audio_file, format_timestamp

logger = logging.getLogger(__name__)

class RealTimeTranscriber:
    def __init__(self, whisper_model, pipeline):
        """Initialize the real-time transcriber"""
        self.whisper_model = whisper_model
        self.pipeline = pipeline
        self.sample_rate = SAMPLE_RATE
        self.audio_data = []
        self.lock = threading.Lock()
        self.is_processing = False
        logger.info("RealTimeTranscriber initialized")

    def add_audio(self, audio_data):
        """Add audio data to the buffer"""
        with self.lock:
            self.audio_data.extend(audio_data)
            logger.debug(f"Added {len(audio_data)} samples to buffer")

    def get_audio_length(self):
        """Get the length of recorded audio in seconds"""
        return len(self.audio_data) / self.sample_rate

    def clear_audio(self):
        """Clear the audio buffer"""
        with self.lock:
            self.audio_data = []

    def process_audio(self, config=None):
        """Process the collected audio data"""
        try:
            if not self.audio_data:
                logger.warning("No audio data to process")
                return None

            self.is_processing = True
            logger.info(f"Processing {self.get_audio_length():.2f} seconds of audio")

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    # Save audio data to file
                    audio_array = np.array(self.audio_data)
                    wavfile.write(tmp_file.name, self.sample_rate, audio_array)
                    logger.info(f"Saved audio to temporary file: {tmp_file.name}")

                    # Process audio using the utility function
                    if config is None:
                        config = {
                            'sample_rate': self.sample_rate,
                            'whisper_language': None,
                            'whisper_task': 'transcribe'
                        }

                    segments, _, _ = process_audio_file(
                        tmp_file.name,
                        self.pipeline,
                        self.whisper_model,
                        config
                    )

                    return segments

                finally:
                    # Cleanup
                    try:
                        os.unlink(tmp_file.name)
                        logger.info("Temporary file cleaned up")
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return None
        finally:
            self.is_processing = False


class AudioRecorder:
    def __init__(self, transcriber):
        """Initialize the audio recorder"""
        self.transcriber = transcriber
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.chunk = CHUNK_SIZE
        self.format = FORMAT
        self.channels = CHANNELS
        self.rate = SAMPLE_RATE
        self.frames = []  # Store raw audio frames
        logger.info("AudioRecorder initialized")

    def callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        try:
            if self.recording:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                if np.any(audio_data):  # Check if there's actual audio data
                    self.frames.append(in_data)  # Store raw frame
                    self.transcriber.add_audio(audio_data)
                    logger.debug(f"Recorded {len(audio_data)} samples")
                else:
                    logger.warning("Empty audio frame received")
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
            return (None, pyaudio.paComplete)

    def start(self):
        """Start recording"""
        try:
            self.recording = True
            self.frames = []  # Clear previous recordings
            self.transcriber.clear_audio()
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.callback
            )
            
            self.stream.start_stream()
            logger.info("Recording started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {str(e)}")
            self.recording = False
            return False

    def stop(self):
        """Stop recording"""
        try:
            if self.recording:
                self.recording = False
                
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                
                self.audio.terminate()
                logger.info("Recording stopped")
                
                # Return the length of recorded audio
                return len(self.frames) * self.chunk / self.rate
                
        except Exception as e:
            logger.error(f"Error stopping recording: {str(e)}")
        finally:
            self.recording = False

    def save_recording(self, filename):
        """Save the recording to a WAV file"""
        try:
            if self.frames:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.rate)
                    wf.writeframes(b''.join(self.frames))
                logger.info(f"Recording saved to {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving recording: {str(e)}")
            return False

    def get_recording_duration(self):
        """Get the current recording duration in seconds"""
        return len(self.frames) * self.chunk / self.rate if self.frames else 0