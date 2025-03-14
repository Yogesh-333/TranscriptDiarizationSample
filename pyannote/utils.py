# utils.py
import logging
import torch
import torchaudio
import numpy as np
import os
import tempfile
from datetime import datetime
import soundfile as sf
import pandas as pd

logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check CUDA and GPU availability"""
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"CUDA {cuda_version} with {gpu_name}"
        return False, "CUDA not available"
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}")
        return False, "Error checking CUDA availability"

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}")
        return "00:00:00"

def save_results(segments, format_type='text'):
    """Save results in specified format"""
    try:
        if format_type == 'csv':
            df = pd.DataFrame(segments)
            return df.to_csv(index=False)
        else:
            text_content = ""
            current_speaker = None
            for segment in sorted(segments, key=lambda x: x['start']):
                if current_speaker != segment['speaker']:
                    text_content += f"\n\n{segment['speaker']}:\n"
                    current_speaker = segment['speaker']
                text_content += f"[{segment['start_time']} - {segment['end_time']}] {segment['text']}\n"
            return text_content
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return None

def process_audio_file(audio_path, pipeline, whisper_model, config):
    """Process audio file with diarization and transcription"""
    try:
        logger.info("Starting audio processing")
        
        # Load audio using soundfile
        audio_data, sr = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
        
        # Resample if necessary
        if sr != 16000:
            audio_tensor = torch.from_numpy(audio_data).float()
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_data = resampler(audio_tensor).numpy()
            sr = 16000
        
        # Perform diarization
        logger.info("Running diarization")
        diarization = pipeline(audio_path)
        
        # Get diarization segments
        diar_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
            logger.info(f"Diarization segment: {speaker} {turn.start:.2f} - {turn.end:.2f}")
        
        if not diar_segments:
            logger.warning("No diarization segments found")
            return [], diarization, None
        
        # Perform transcription
        logger.info("Running transcription")
        transcription = whisper_model.transcribe(
            audio_data,
            language=config.get('whisper_language'),
            task=config.get('whisper_task', 'transcribe')
        )
        
        for seg in transcription['segments']:
            logger.info(f"Transcription segment: {seg['start']:.2f} - {seg['end']:.2f}: {seg['text']}")
        
        if not transcription['segments']:
            logger.warning("No transcription segments found")
            return [], diarization, transcription
        
        # Match transcription with speakers
        logger.info("Matching transcription with speakers")
        final_segments = []
        
        for trans_segment in transcription['segments']:
            segment_start = trans_segment['start']
            segment_end = trans_segment['end']
            
            # Find all speakers during this segment
            segment_speakers = {}
            
            for diar_seg in diar_segments:
                # Calculate overlap
                if segment_end > diar_seg['start'] and segment_start < diar_seg['end']:
                    overlap_start = max(segment_start, diar_seg['start'])
                    overlap_end = min(segment_end, diar_seg['end'])
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration > 0:
                        segment_speakers[diar_seg['speaker']] = segment_speakers.get(diar_seg['speaker'], 0) + overlap_duration
            
            # Assign to speaker with most overlap
            if segment_speakers:
                dominant_speaker = max(segment_speakers.items(), key=lambda x: x[1])[0]
                
                final_segments.append({
                    'speaker': dominant_speaker,
                    'start': segment_start,
                    'end': segment_end,
                    'start_time': format_timestamp(segment_start),
                    'end_time': format_timestamp(segment_end),
                    'text': trans_segment['text'].strip()
                })
                
                logger.info(f"Matched segment: {dominant_speaker} {segment_start:.2f} - {segment_end:.2f}: {trans_segment['text']}")
        
        if not final_segments:
            logger.warning("No segments were produced after matching")
        else:
            logger.info(f"Successfully created {len(final_segments)} segments")
        
        return final_segments, diarization, transcription
        
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        raise

def initialize_models(hf_token, whisper_model_size, use_gpu):
    """Initialize Whisper and Pyannote models"""
    try:
        from pyannote.audio import Pipeline
        import whisper
        
        # Initialize pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if use_gpu and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        
        # Initialize Whisper
        whisper_model = whisper.load_model(whisper_model_size)
        if use_gpu and torch.cuda.is_available():
            whisper_model = whisper_model.to(torch.device("cuda"))
        
        logger.info("Models initialized successfully")
        return pipeline, whisper_model
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise