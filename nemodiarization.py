import streamlit as st
import torch
import whisper
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.diarization_utils import ASRDiarization
import os
import tempfile

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load NeMo speaker diarization model
@st.cache_resource
def load_nemo_model():
    return EncDecSpeakerLabelModel.from_pretrained("titanet_large")

# Function to perform diarization
def diarize_audio(audio_path, whisper_model, nemo_model):
    # Transcribe audio using Whisper
    result = whisper_model.transcribe(audio_path)
    transcription = result['text']

    # Perform speaker diarization using NeMo
    diarization = ASRDiarization(nemo_model)
    diarization_result = diarization.diarize(audio_path)

    return transcription, diarization_result

# Streamlit UI
st.title("Audio Diarization with Whisper and NeMo")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    # Load models
    whisper_model = load_whisper_model()
    nemo_model = load_nemo_model()

    # Perform diarization
    transcription, diarization_result = diarize_audio(audio_path, whisper_model, nemo_model)

    # Display results
    st.subheader("Transcription")
    st.write(transcription)

    st.subheader("Diarization Result")
    st.write(diarization_result)

    # Clean up the temporary file
    os.unlink(audio_path)