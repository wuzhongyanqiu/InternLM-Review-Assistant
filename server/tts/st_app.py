import uuid
import requests
import time
import json
import streamlit as st
import base64
import numpy as np

def get_tts_api(req_data: dict):
    # 获取 TTS 结果
    res = requests.post(f"http://0.0.0.0:8002/generate_audio", json=req_data).json()
    audio_data_bytes = base64.b64decode(res['audio_data_base64'])
    sample_rate = res['sample_rate']
    audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.float32)

    return audio_data_np, sample_rate 

def get_tts_reqdata(text_input, temperature, top_P, top_K, refine_text_flag=True):
    req_data = {
        'text': str(text_input),
        'temperature': float(temperature),
        'top_P': float(top_P),
        'top_K': int(top_K),
        'refine_text_flag': bool(refine_text_flag)
    }
    return req_data

# 在 Streamlit 应用中
def main():
    # 设置页面标题
    st.title("ChatTTS Demo")

    # 默认文本和文本输入框
    default_text = "你好，你可以详细介绍一下DeepSpeed Sparse Attention吗？ "
    text_input = st.text_input("Input Text", value=default_text)

    # 行内布局和滑动条
    with st.expander("Advanced Settings"):
        temperature = st.slider("Audio temperature", min_value=0.00001, max_value=1.0, value=0.5)
        top_P = st.slider("top_P", min_value=0.1, max_value=0.9, step=0.05, value=0.5)
        top_K = st.slider("top_K", min_value=1, max_value=20, step=1, value=20)

    # 生成按钮
    generate_button = st.button("Generate")

    # 输出文本和音频
    # st.text_area("Output Text", height=300, key="-1", disabled=True)
    # st.audio("", format="audio")

    if generate_button:
        # 调用 generate_audio 函数生成音频
        req_data = get_tts_reqdata(text_input, temperature, top_P, top_K, True)
        print(req_data)
        audio_data_np, sample_rate = get_tts_api(req_data)

        # 使用 st.audio 播放音频
        st.audio(audio_data_np, sample_rate=sample_rate)

# 运行 Streamlit 应用
if __name__ == "__main__":
    main()
    
