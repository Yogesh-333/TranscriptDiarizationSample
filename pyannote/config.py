import pyaudio

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# UI Configuration
PAGE_TITLE = "Audio Transcription & Speaker Diarization"
PAGE_ICON = "🎤"
LAYOUT = "wide"

# CSS Styles
CUSTOM_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin: 1em 0;
    }
    .speaker-header {
        color: #1f77b4;
        margin-top: 20px;
        font-weight: bold;
    }
    .timestamp {
        color: #666;
        font-size: 0.9em;
        font-family: monospace;
    }
    .transcription-text {
        font-size: 1.1em;
        margin: 5px 0;
        padding: 5px;
        border-left: 3px solid #1f77b4;
    }
    .recording-active {
        color: red;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    </style>
"""

# Model Configuration
DEFAULT_WHISPER_MODEL = "base"
SUPPORTED_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
SUPPORTED_AUDIO_TYPES = ['wav', 'mp3']