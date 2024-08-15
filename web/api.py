import uuid
import requests
import time

def get_asr_api(audio_path):
    # 获取 ASR 结果
    req_data = {
        "request_id": str(uuid.uuid1()),
        "audio_path": audio_path,
    }

    print(req_data)

    res = requests.post(f"http://0.0.0.0:8001/asr", json=req_data).json()
    return res["result"]

def get_tts_api(tts_text, tts_path):
    # 获取 TTS 结果
    req_data = {
        "request_id": str(uuid.uuid1()),
        "tts_text": tts_text,
        "tts_path": tts_path,
    }

    print(req_data)
    res = requests.post(f"http://0.0.0.0:8002/tts", json=req_data).json()
    return res["result"]

def get_streamchat_responses(prompt):
    req_data = {
        "inputs": prompt
    }
    res = requests.post(f"http://0.0.0.0:8003/streamchat", json=req_data, stream=True)
    for line in res.iter_lines():
        if line:
            yield line.decode('utf-8')
            # time.sleep(1)

def get_chat_responses(prompt):
    req_data = {
        "inputs": prompt
    }
    res = requests.post(f"http://0.0.0.0:8003/chat", json=req_data, stream=True)
    return res.response.text

def get_selectquestion():
    req_data = {
        "toolname": 'selectquestiontool',
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

def get_answerevaluation(query, ans):
    req_data = {
        "toolname": "answerevaluationtool",
        "query": query,
        "ans": ans
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

def get_parsingresumes(document_path):
    req_data = {
        "toolname": "parsingresumestool",
        "document_path": document_path
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

