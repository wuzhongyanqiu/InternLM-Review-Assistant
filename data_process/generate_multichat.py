import sys
sys.path.append('../')  # 添加父目录到系统路径
from openai import OpenAI
import os
import erniebot
from agent.agent_tools import ParsingResumesTool
import json

parsingresumestool = ParsingResumesTool()
document_path = "/root/Mock-Interviewer/agent/upload_resume.pdf"
result = parsingresumestool.reply(document_path).strip().replace("xxx", "张三").replace("159-1075-7403", "111-1111-1111")

##################################################################################
#########################通义千问##################################################
##################################################################################
def get_response(messages):
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key="YOUR API KEY", 
        # 填写DashScope服务的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
        temperature=0.8,
        top_p=0.8
        )
    return completion


##################################################################################
#########################文心一言##################################################
##################################################################################

erniebot.api_type = "API_TYPE"
erniebot.access_token = "API_TOKEN"

# def gen_wenxin_messages(prompt):
#     messages = [{"role": "user", "content": prompt}]
#     return messages


def get_completion(messages, model="ernie-3.5", temperature=0.5, system_prompt_ernie=""):
    chat_comp = erniebot.ChatCompletion()

    resp = chat_comp.create(messages=messages, 
                        model=model,
                        temperature = temperature,
                        system=system_prompt_ernie)

    return resp["result"]

if __name__ == "__main__":
    conversations = []
    for i in range(50):
        conversation = {}
        conversation['conversation'] = []
        resume_prompt = [{'role': 'user', 'content': "阅读这段简历的全部内容：\n{}\n提取出关键的信息，整理为一个简洁的文本并输出，不超过300字".format(result)}]
        resume_text = get_response(resume_prompt).choices[0].message.content
        # print(resume_text)
        system_prompt_qwen = "你是一个面试官，你能根据面试者的简历信息和对面试者进行面试，你一次最多提出一个问题，你的问题必须与简历内容相关，注意，如果简历中有一些专业名词，你可以直接问面试者相关内容，面试者的每个回答你都会给予反馈，然后追问或继续抛出问题，你的话语要简洁。"
        system_prompt_ernie = "你是一个面试者，你将根据历史对话内容回答面试官的问题，作为一个初出茅庐的校招生，你的回答不能过于完美，同时要注重简洁明了，不超过300字。"
        qwen_messages = [{'role': 'system', 'content': system_prompt_qwen}]
        wenxin_messages = []
        init_input = "您好，面试官，我的简历内容是：\n{}\n，请开始面试".format(resume_text)
        print(init_input)
        qwen_messages.append({'role': 'user', 'content': init_input})
        assistant_output = get_response(qwen_messages).choices[0].message.content
        print(assistant_output)
        qwen_messages.append({'role': 'assistant', 'content': assistant_output})
        wenxin_messages.append({'role': 'user', 'content': assistant_output})
        tmp_dict_outside = {
            "system": system_prompt_qwen,
            "input": init_input,
            "output": assistant_output
        }
        conversation['conversation'].append(tmp_dict_outside)
        for i in range(2):
            user_input = get_completion(messages=wenxin_messages, system_prompt_ernie=system_prompt_ernie)
            print(user_input)
            qwen_messages.append({'role': 'user', 'content': user_input})
            wenxin_messages.append({'role': 'assistant', 'content': user_input})
            assistant_output = get_response(qwen_messages).choices[0].message.content
            print(assistant_output)
            # 将大模型的回复信息添加到messages列表中
            qwen_messages.append({'role': 'assistant', 'content': assistant_output})
            wenxin_messages.append({'role': 'user', 'content': assistant_output})
            tmp_dict_inside = {
                "input": user_input,
                "output": assistant_output
            }
            conversation['conversation'].append(tmp_dict_inside)

        conversations.append(conversation)
    with open('../datas/multi_interview_data1.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
        
        # print(qwen_messages)
        # print("################################################################\n#########################################")
        # print(wenxin_messages)