import gradio as gr
import soundfile as sf
import time
import torch
import numpy as np
import logging
import librosa
from faster_whisper import WhisperModel

LINE_HEIGHT = 27

assert torch.cuda.is_available(), "Please change your runtime type to GPU to enable fast generation."

cool_whisper_model_card = "andybi7676/cool-whisper" # model card, will download the model form Hugging Face
cool_whisper_model = WhisperModel(cool_whisper_model_card, device="cuda", compute_type="float16")

whisper_model_card = "large-v2" # model card, will download the model form Hugging Face
whisper_model = WhisperModel(whisper_model_card, device="cuda", compute_type="float32")
# model = WhisperModel(model_card, device="cpu", compute_type="float32")

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logger.level)

# Function to process the audio input
def process_audio_to_cool_whisper(audio_fpath):
    # print(audio)
    # print(type(audio))
    audio_fpath = "./test.wav"
    audio_info = sf.info(audio_fpath)
    print(audio_info) # for debug

    start_time = time.time()
    segments, info = cool_whisper_model.transcribe(audio_fpath, beam_size=5, language="zh", condition_on_previous_text=True, vad_filter=True) # zh for zh-en code-switching in k2d-whisper
    current_cool_whisper_transcript = ""
    for segment in segments:
        new_transcript = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(new_transcript)
        current_cool_whisper_transcript += new_transcript + "\n"
        yield current_cool_whisper_transcript
    end_time = time.time()
    print(f"Total processing time: {end_time -  start_time:.2f}") # calculate the overall processing time
    current_cool_whisper_transcript += f"Total processing time: {end_time -  start_time:.2f}"
    yield current_cool_whisper_transcript
    # whisper large v2
    
def process_audio_to_whisper_large(audio_fpath):
    audio_fpath = "./test.wav"
    audio_info = sf.info(audio_fpath)
    print(audio_info) # for debug
    start_time = time.time()
    segments, info = whisper_model.transcribe(audio_fpath, beam_size=5, language="zh", condition_on_previous_text=True) # zh for zh-en code-switching in k2d-whisper
    current_whisper_transcript = ""
    for segment in segments:
        new_transcript = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(new_transcript)
        current_whisper_transcript += new_transcript + "\n"
        yield current_whisper_transcript
    end_time = time.time()
    print(f"Total processing time: {end_time -  start_time:.2f}") # calculate the overall processing time
    current_whisper_transcript += f"Total processing time: {end_time -  start_time:.2f}"
    yield current_whisper_transcript

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
    submit_btn.click(fn=process_audio_to_whisper_large, inputs=audio_input, outputs=text2)

# Launching the interface
demo.launch()
