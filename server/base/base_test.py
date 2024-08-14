import requests
messages = [{'role': 'user', 'content': 'who are you?'}]
req_data = {
    "inputs": messages
}
res = requests.post(f"http://0.0.0.0:8003/streamchat", json=req_data, stream=True)
for line in res.iter_lines():
    if line:
        print(line.decode('utf-8'))
# print(line.decode('utf-8'))
# print(res)
# for cur_res in res:
#     print(cur_res)