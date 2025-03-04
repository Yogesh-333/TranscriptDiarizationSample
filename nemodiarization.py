import gradio as gr
import subprocess
import os
import shutil
import venv

# --- Configuration ---
whisper_diarization_path = "/path/to/your/whisper-diarization"  # **REPLACE with the actual path**
venv_dir = "venv_diarization"  # Name of the virtual environment directory

# --- Virtual Environment Management ---
def create_or_activate_venv():
    """Creates or activates a virtual environment."""
    venv_path = os.path.join(os.getcwd(), venv_dir)

    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        print("Installing required packages...")
        pip_path = os.path.join(venv_path, "bin", "pip")  # Linux/macOS
        if os.name == "nt":  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        subprocess.run([pip_path, "install", "gradio", "whisper-diarization", "ffmpeg-python"]) # Install in venv

    # Activate (for demonstration, not strictly needed for subprocess)
    if os.name == "nt":
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        os.system(f"cmd /c {activate_script}")
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
        os.system(f"source {activate_script}")


# --- Diarization Logic ---
def diarize_audio(audio_file, model_size):
    try:
        temp_dir = "temp_diarization"
        os.makedirs(temp_dir, exist_ok=True)

        audio_path = os.path.join(temp_dir, audio_file.name)
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        python_executable = os.path.join(venv_dir, "bin", "python")  # Linux/macOS
        if os.name == "nt":
            python_executable = os.path.join(venv_dir, "Scripts", "python.exe")

        command = [
            python_executable,
            os.path.join(whisper_diarization_path, "main.py"),
            "--model", model_size,
            "--audio", audio_path,
            "--output_rttm", os.path.join(temp_dir, "output.rttm")
        ]

        process = subprocess.Popen(command, cwd=whisper_diarization_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stderr:
            return None, stderr.decode()

        rttm_path = os.path.join(temp_dir, "output.rttm")
        return rttm_path, None

    except Exception as e:
        return None, str(e)


def display_rttm(rttm_path):
    if rttm_path:
        with open(rttm_path, "r") as f:
            return f.read()
    return "No RTTM file generated."


# --- Gradio Interface ---
with gr.Blocks() as demo:
    create_or_activate_venv()  # Create/activate venv

    gr.Markdown("## Whisper Diarization Demo")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        model_size = gr.Dropdown(["tiny", "base", "small", "medium", "large"], label="Model Size", value="small")

    diarize_button = gr.Button("Diarize")
    rttm_output = gr.Textbox(label="RTTM Output", lines=10)

    diarize_button.click(diarize_audio, inputs=[audio_input, model_size], outputs=[rttm_output, rttm_output])

demo.launch()