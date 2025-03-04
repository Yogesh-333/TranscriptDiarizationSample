import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import librosa
import soundfile as sf
import numpy as np

class DiarizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Diarization")

        self.audio_file = None
        self.output_file = None

        self.label_audio = tk.Label(master, text="Audio File:")
        self.label_audio.grid(row=0, column=0, sticky="w")

        self.entry_audio = tk.Entry(master, width=50)
        self.entry_audio.grid(row=0, column=1, padx=5)

        self.button_browse_audio = tk.Button(master, text="Browse", command=self.browse_audio)
        self.button_browse_audio.grid(row=0, column=2)

        self.button_diarize = tk.Button(master, text="Diarize", command=self.diarize_audio, state=tk.DISABLED)
        self.button_diarize.grid(row=1, column=1, pady=10)


    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.ogg")])
        if self.audio_file:
            self.entry_audio.delete(0, tk.END)
            self.entry_audio.insert(0, self.audio_file)
            self.button_diarize.config(state=tk.NORMAL)  # Enable diarization button


    def diarize_audio(self):
        try:
            # 1. Resample audio to 16kHz if necessary (UIS-RNN requirement)
            y, sr = librosa.load(self.audio_file, sr=None)
            if sr != 16000:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
                temp_audio = "temp_16kHz.wav"
                sf.write(temp_audio, y_resampled, 16000)
                audio_input = temp_audio
            else:
                audio_input = self.audio_file

            # 2. Run UIS-RNN diarization
            self.output_file = os.path.splitext(self.audio_file)[0] + "_diarized.rttm"
            command = [
                "python", "uisrnn/speaker_diarization.py",  # Path to your speaker_diarization.py
                "--input_wave", audio_input,
                "--output_rttm", self.output_file
            ]
            subprocess.run(command, check=True)  # Use check=True to raise an exception on error

            if sr != 16000:  # Remove temporary file
                os.remove(temp_audio)

            messagebox.showinfo("Diarization Complete", f"Diarization complete. RTTM file saved to:\n{self.output_file}")

        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Diarization failed:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")



root = tk.Tk()
app = DiarizationApp(root)
root.mainloop()