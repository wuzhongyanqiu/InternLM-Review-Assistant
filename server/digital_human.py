import re
import os

def extract_last_path(output: str) -> str:
    # 使用正则表达式匹配所有.mp4路径
    paths = re.findall(r'[\w./]+\.mp4', output)
    
    # 如果找到路径，返回最后一个
    if paths:
        return os.path.abspath(paths[-1])
    else:
        return None

def gen_digital_human(driven_audio: str, source_image: str):
    import requests

    url = 'http://127.0.0.1:8006/inference/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = {
        "driven_audio": driven_audio,
        "source_image": source_image
    }

    response = requests.post(url, json=data, headers=headers)

    print(response.status_code)
    return extract_last_path(response.json()['output'])

