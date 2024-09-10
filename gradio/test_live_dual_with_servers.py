import gradio as gr
import soundfile as sf
import time
import torch
import numpy as np
import requests
import os
from dotenv import load_dotenv

WORKDIR = os.getenv("WORKDIR")
ENV = load_dotenv(f"{WORKDIR}/.env")
SERVER_PORT = ENV["SERVER_PORT"]

LINE_HEIGHT = 27

assert torch.cuda.is_available(), "Please change your runtime type to GPU to enable fast generation."

# cool_whisper_model_card = "andybi7676/cool-whisper" # model card, will download the model form Hugging Face
# cool_whisper_model = WhisperModel(cool_whisper_model_card, device="cuda", compute_type="float16")

# whisper_model_card = "large-v2" # model card, will download the model form Hugging Face
# whisper_model = WhisperModel(whisper_model_card, device="cuda", compute_type="float32")
# # model = WhisperModel(model_card, device="cpu", compute_type="float32")

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)
# logging.getLogger("faster_whisper").setLevel(logger.level)

# Function to process the audio input
def process_audio_to_cool_whisper(audio_fpath, server_ip=ENV["COOL_WHISPER_SERVER_IP"]):
    # print(audio)
    # print(type(audio))
    audio_fpath = f"{WORKDIR}/gradio/.local/test.wav"
    audio_info = sf.info(audio_fpath)
    print(audio_info) # for debug

    if audio_fpath is not None:
        files = {'audio': open(audio_fpath, 'rb')}
        
        # Stream the response
        with requests.post(f"http://{server_ip}:8000/upload_audio_and_transcribe", files=files, stream=True) as response:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:  # Filter out keep-alive new chunks
                    chunk_text = chunk.decode('utf-8')
                    print(chunk_text, end="")
                    response_text += chunk_text + "\n"
                    yield response_text  # Gradually update the text box

    
def process_audio_to_whisper_large(audio_fpath, server_ip=ENV["WHISPER_LARGE_SERVER_IP"]):

    audio_fpath = f"{WORKDIR}/gradio/.local/test.wav"
    audio_info = sf.info(audio_fpath)
    print(audio_info) # for debug

    if audio_fpath is not None:
        files = {'audio': open(audio_fpath, 'rb')}
        
        # Stream the response
        with requests.post(f"http://{server_ip}:8000/upload_audio_and_transcribe", files=files, stream=True) as response:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:  # Filter out keep-alive new chunks
                    chunk_text = chunk.decode('utf-8')
                    print(chunk_text, end="")
                    response_text += chunk_text + "\n"
                    yield response_text  # Gradually update the text box

# Creating the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        text1 = gr.Textbox(label="Cool-Whisper", lines=LINE_HEIGHT)
        text2 = gr.Textbox(label="Whisper-large-v2", lines=LINE_HEIGHT)
    
    with gr.Row():
        audio_input = gr.Audio(label="Audio Input", sources=["microphone", "upload"], type="filepath")

    # Button to submit
    submit_btn = gr.Button("Submit")
    
    # Linking button to audio processing function
    submit_btn.click(fn=process_audio_to_cool_whisper, inputs=audio_input, outputs=text1)
    # submit_btn.click(fn=process_audio_to_whisper_large, inputs=audio_input, outputs=text2)

# Launching the interface
demo.launch()
