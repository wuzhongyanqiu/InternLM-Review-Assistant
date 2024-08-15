import torch
from TTS.api import TTS

def load_tts_model():
    model = TTS(model_path="/root/Mock-Interviewer/XTTS-v2", config_path="/root/Mock-Interviewer/XTTS-v2/config.json")
    return model

def save_tts_text(text, model, tts_path):
    model.tts_to_file(text=text, speaker_wav="/root/Mock-Interviewer/audio/my_audio.wav", language="zh-cn", file_path=tts_path)
    return tts_path

