import os
import random
import argparse

import torch
import streamlit as st
import numpy as np

import server.tts.ChatTTS as ChatTTS
import torchaudio

MODEL_PATH = "/root/Mock-Interviewer/models/ChatTTS"

print("loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load_models(local_path="/root/Mock-Interviewer/models/ChatTTS")

def generate_audio(text, temperature, top_P, top_K, refine_text_flag):
    audio_seed = 42
    text_seed = 42
    torch.manual_seed(audio_seed)
    rand_spk = torch.randn(768)
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed)

    if refine_text_flag:
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000

    return audio_data, sample_rate