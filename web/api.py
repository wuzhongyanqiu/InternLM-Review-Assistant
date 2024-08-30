import uuid
import requests
import time
import json

def get_asr_api(audio_path):
    # 获取 ASR 结果
    req_data = {
        "request_id": str(uuid.uuid1()),
        "audio_path": audio_path,
    }

    print(req_data)

    res = requests.post(f"http://0.0.0.0:8001/asr", json=req_data).json()
    return res["result"]

def get_tts_api(req_data: dict):
    # 获取 TTS 结果
    res = requests.post(f"http://0.0.0.0:8002/generate_audio", json=req_data).json()
    return res['audio_data'], res['sample_rate'] 

def get_selectquestion():
    req_data = {
        "toolname": 'selectquestiontool',
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

def get_answerevaluation(query, ans, rag_content):
    req_data = {
        "toolname": "answerevaluationtool",
        "query": query,
        "ans": ans,
        "rag_content": rag_content
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

def gen_database():
    try:
        response = requests.post("http://0.0.0.0:8004/tools/gen_database")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return False
    return True
    
def get_parsingresumes(document_path):
    req_data = {
        "toolname": "parsingresumestool",
        "document_path": document_path
    }
    res = requests.post(f"http://0.0.0.0:8004/tools", json=req_data).json()
    return res['result']

if __name__ == "__main__":
    messages = [{'role': 'system', 'content': '你是一个友善的AI助手'}, {'role': 'user', 'content': '你好'}]
    result = get_streamchat_responses(messages)
    print(result)
