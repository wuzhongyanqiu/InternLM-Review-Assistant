################ openai 调用 lmdeploy server ##################

from openai import OpenAI

client = OpenAI(
    api_key='YOUR_API_KEY',  
    base_url="http://0.0.0.0:23333/v1"  
)

model_name = client.models.list().data[0].id

def chat(messages):
    response = client.chat.completions.create(
        model=model_name,  
        messages=messages,
        temperature=0.8,  
        top_p=0.8  
    )

    return response.choices[0].message.content

############### APIClient 调用 lmdeploy server ################

from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient(f'http://0.0.0.0:23333')
messages = [
    "hi, what's your name?",
    "who developed you?",
    "Tell me more about your developers",
    "Summarize the information we've talked so far"
]
for message in messages:
    for item in api_client.chat_interactive_v1(prompt=message,
                                               session_id=1,
                                               interactive_mode=True,
                                               stream=False):
        print(item)





