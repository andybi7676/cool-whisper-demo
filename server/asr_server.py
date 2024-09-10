from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
import logging
import torch
import time
import argparse
import json
import os
from dotenv import dotenv_values
from faster_whisper import WhisperModel

WORKDIR = os.getenv("WORKDIR")
ENV = dotenv_values(f"{WORKDIR}/.env")

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_card", type=str, default="andybi7676/cool-whisper", help="The model card to use for the ASR model. Options: [andybi7676/cool-whisper, large-v2]")

args = argparser.parse_args()

model_card = args.model_card
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
model = WhisperModel(model_card, device=device, compute_type=compute_type)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logging.getLogger("faster_whisper").setLevel(logger.level)

logger.info(f"Model loaded successfully. | model card: {model_card}, device: {device}, compute type: {compute_type}")

app = FastAPI()

async def fake_json_streamer():
    t0 = time.time()
    for i in range(10):
        print(f"Chunk being yielded (time {int((time.time()-t0)*1000)}ms)", flush=True)
        yield json.dumps( {"message": "Hello World"}) + '\n'
        time.sleep(0.5)
    print(f"Over (time {int((time.time()-t0)*1000)}ms)", flush=True)

@app.get("/test")
async def test():
    return StreamingResponse(fake_json_streamer(), media_type='text/event-stream')

# Streaming response endpoint
@app.post("/upload_audio_and_transcribe")
async def upload_audio_and_transcribe(audio: UploadFile = File(...)):
    audio_fpath = f"./.local/{audio.filename}"
    
    # Save the received audio file
    with open(audio_fpath, "wb") as file:
        file.write(await audio.read())

    # Streaming the response back to the client
    def response_generator():
        start_time = time.time()
        segments, info = model.transcribe(audio_fpath, beam_size=5, language="zh", condition_on_previous_text=True, vad_filter=True) # zh for zh-en code-switching in k2d-whisper
        # current_cool_whisper_transcript = ""
        for segment in segments:
            new_transcript = "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
            print(new_transcript, end="")
            # current_cool_whisper_transcript += new_transcript + "\n"
            yield new_transcript
        end_time = time.time()
        print(f"Total processing time: {end_time -  start_time:.2f}") # calculate the overall processing time
        yield f"Processing complete!\tTotal processing time: {end_time -  start_time:.2f}\n"

    # return StreamingResponse(response_generator(), media_type="text/plain")
    return StreamingResponse(response_generator(), media_type="text/event-stream")

# if __name__ == "__main__":

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=int(ENV["SERVER_PORT"]))