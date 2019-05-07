import _init_path
from models.wav2letter import Wav2letter
import sys
import json
import wave
from pyaudio import PyAudio, paInt16


model = Wav2letter.load("pretrained/wav2letter.pth")

framerate = 16000
NUM_SAMPLES = 2000
channels = 1
sampwidth = 2
TIME = 10


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()


def record():
    pa = PyAudio()
    stream = pa.open(
        format=paInt16,
        channels=1,
        rate=framerate,
        input=True,
        frames_per_buffer=NUM_SAMPLES,
    )
    my_buf = []
    count = 0
    print("录音中(5s)")
    while count < TIME * 5:
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count += 1
        print(".", end="", flush=True)

    save_wave_file("01.wav", my_buf)
    stream.close()


record()

text = model.predict("01.wav")

print("")
print("识别结果:")
print(text)
