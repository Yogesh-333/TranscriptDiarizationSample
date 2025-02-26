import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['SPEECHBRAIN_DISABLE_SYMLINKS'] = '1'

import whisper
import torch
from pyannote.audio import Pipeline
import datetime
import wave
import contextlib
from huggingface_hub import login
from pydub import AudioSegment
import soundfile as sf

class AudioTranscriber:
    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.whisper_model = None
        self.pipeline = None
        
        # Install required packages
        os.system('pip install pydub soundfile')
        
        try:
            login(auth_token)
            print("Successfully logged in to Hugging Face")
        except Exception as e:
            print(f"Error logging in to Hugging Face: {str(e)}")

    def convert_to_wav(self, input_file):
        """Convert audio file to WAV format"""
        try:
            print(f"Converting {input_file} to WAV format...")
            # Get the file extension
            file_extension = os.path.splitext(input_file)[1].lower()
            
            if file_extension == '.wav':
                # Check if it's a valid WAV file
                try:
                    with wave.open(input_file, 'rb') as wave_file:
                        return input_file
                except:
                    print("Invalid WAV file, attempting to convert...")
            
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(input_file)
            wav_path = os.path.splitext(input_file)[0] + '_converted.wav'
            audio.export(wav_path, format='wav')
            print(f"Converted file saved as: {wav_path}")
            return wav_path
            
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            return None

    def validate_audio_file(self, file_path):
        """Validate audio file"""
        try:
            # Try reading with soundfile
            data, samplerate = sf.read(file_path)
            return True
        except Exception as e:
            print(f"Invalid audio file: {str(e)}")
            return False

    def transcribe_audio(self, audio_path):
        try:
            # Convert audio to WAV if needed
            wav_file = self.convert_to_wav(audio_path)
            if not wav_file:
                raise Exception("Failed to convert audio file")

            # Validate audio file
            if not self.validate_audio_file(wav_file):
                raise Exception("Invalid audio file")

            if not self.whisper_model or not self.pipeline:
                self.initialize_models()

            print("Starting transcription...")
            result = self.whisper_model.transcribe(wav_file)
            
            print("Performing speaker diarization...")
            diarization = self.pipeline(wav_file)

            segments = []
            for segment, track, label in diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                
                current_text = ""
                for item in result["segments"]:
                    if (item["start"] >= start_time and item["start"] < end_time) or \
                       (item["end"] > start_time and item["end"] <= end_time):
                        current_text += item["text"] + " "
                
                if current_text.strip() != "":
                    segments.append({
                        "speaker": label,
                        "start": start_time,
                        "end": end_time,
                        "text": current_text.strip()
                    })

            # Clean up converted file if it was created
            if wav_file != audio_path and os.path.exists(wav_file):
                os.remove(wav_file)

            return segments

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def initialize_models(self):
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("small")
            
            print("Loading Speaker Diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.auth_token
            )
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def format_timestamp(self, seconds):
        return str(datetime.timedelta(seconds=round(seconds)))

def main():
    # Replace with your actual values
    HF_TOKEN = "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"
    AUDIO_FILE = "D:\DOWNLOAD\Doctor-Patient Cost of Care Conversation.mp3"  # Can be MP3, WAV, M4A, etc.
    OUTPUT_FILE = "transcription.txt"

    try:
        # Install required packages
        os.system('pip install pydub soundfile')
        
        transcriber = AudioTranscriber(HF_TOKEN)
        print(f"Processing audio file: {AUDIO_FILE}")
        segments = transcriber.transcribe_audio(AUDIO_FILE)

        if segments:
            print("\nTranscription Results:")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for segment in segments:
                    output = f"\nSpeaker: {segment['speaker']}\n"
                    output += f"Time: {transcriber.format_timestamp(segment['start'])} -> "
                    output += f"{transcriber.format_timestamp(segment['end'])}\n"
                    output += f"Text: {segment['text']}\n"
                    
                    print(output)
                    f.write(output)
            
            print(f"\nTranscription saved to {OUTPUT_FILE}")
        else:
            print("Transcription failed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()