from openai import OpenAI
from typing import List, Union
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:8005/v1')
model_name = client.models.list().data[0].id

def get_image_chat(image_url: Union[str, List[str]]):
    responses = []
    if isinstance(image_url, str):
        images_url = [image_url]
    elif isinstance(image_url, list):
        images_url = image_url
    else:
        raise TypeError("image_url must be a string or a list of string")

    for item in images_url:
        response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': '<image>\n将图里的信息提取出几道面试题，每道面试题必须以<|start|>开始，并且以<|end|>结束',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url':
                    item,
                },
            }],
        }],
        temperature=0.8,
        top_p=0.8)
        responses.append(response.choices[0].message.content)
    return responses

if __name__ == "__main__":
    response = get_image_chat('/root/InternLM-Interview-Assistant/datas/pics/pictest.png')
    print(response)
