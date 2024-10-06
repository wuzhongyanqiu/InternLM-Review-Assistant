import requests
import json

def rag_gendatabase():
    url = 'http://127.0.0.1:8004/rag/gendb'
    response = requests.post(url)

    if response.status_code == 200:
        print("成功生成问题数据库&RAG知识库")
        print("响应内容:", response.json())
    else:
        print("请求失败，状态码:", response.status_code)
        print("错误信息:", response.text)
    
def rag_comments(query, ans):
    url = 'http://127.0.0.1:8004/rag/comments'

    rag_item = {
        "query": query,
        "ans": ans
    }

    json_data = json.dumps(rag_item)

    response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

def rag_question(chat_content):
    url = 'http://127.0.0.1:8004/rag/questions'

    rag_item = {
        "chat_content": chat_content,
    }

    json_data = json.dumps(rag_item)

    response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

if __name__ == '__main__':
    rag_gendatabase()
    rag_question("我对CUDA的程序进行了一系列优化，这让我节省了资源和加快了速度")