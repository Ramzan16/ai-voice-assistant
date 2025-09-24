from kokoro import KModel, KPipeline  # type: ignore


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

import torch
import sounddevice as sd  # type: ignore
import numpy as np
from time import time


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=api_key
)

stream = model.stream("Write code for a function that computes the nth Fibonacci number.")
# stream = model.invoke("Tell me a story about a person stuck in outer space.")


print("Streaming response:")
for chunk in stream:
    print(chunk.content, end="", flush=True)


# pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a', device='cpu')
# friday = 'af_bella'
# samplerate = 24000
# text_to_speak = stream.content

# print("\nGenerating speech...")

# stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype='float32')
# stream.start()
# start = time()

# for result in pipeline(text=text_to_speak, voice=friday):
#     print(f"Generated audio for text chunk: '{result.graphemes[:40]}...'")
#     if result.audio is not None:
#         chunk = result.audio.detach().cpu().numpy().astype('float32')
#         stream.write(chunk)

# stream.stop()
# end = time()
# print(f"\nTime taken: {end - start:.2f} seconds")
# stream.close()
# print("\nStreaming finished.")