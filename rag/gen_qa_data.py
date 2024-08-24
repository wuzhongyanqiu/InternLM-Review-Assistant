import re
import json
import sqlite3
from zhipuai import ZhipuAI
import os 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

ZHIPUAI_API_KEY = os.environ['ZHIPUAI_API_KEY']
client = ZhipuAI(
    api_key=ZHIPUAI_API_KEY
)

system_prompt = """
你是一个面试官，你擅长将给定内容改写成一个面试题和其对应的答案。\n
要求，改写的面试题必须以给定内容相关。\n
注意，你的答案要求简洁明了
输出格式：\n
<question_start>你的面试题<question_end>
<answer_start>对应的答案<answer_end>
"""

def gen_glm_params(prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": "你要改写的问题是{}".format(prompt)})
    return messages

def get_completion(prompt, model="glm-4", temperature=0.95):
    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

# 打开文件并读取内容
def read_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

def find_headers(markdown_content):
    second_level_headers = []
    second_level_contents = []
    pattern = re.compile(r'^##\s*(.*?)(?=\n|#|$)', re.MULTILINE)
    for match in pattern.finditer(markdown_content):
        second_level_headers.append(match.group(1))
        content = ''
        try:
            next_start = next(m.start() for m in pattern.finditer(markdown_content) if m.start() > match.end())
            content += markdown_content[match.end():next_start].strip()
        except StopIteration:
            content += markdown_content[match.end():].strip()

        second_level_contents.append(content.strip())
        last_end = match.end()
    
    return second_level_headers, second_level_contents

if __name__ == '__main__':
    # 调用函数，传入Markdown文件的路径
    file_path = '../datas/rag_test_qa.md'
    datas = []
    markdown_content = read_markdown_file(file_path)

    second_level_headers, second_level_contents = find_headers(markdown_content)

    for i, header in enumerate(second_level_headers):
        qa_content = get_completion(header + '\n' + second_level_contents[i])
        pattern_q = re.compile(r"<question_start>(.*?)<question_end>", re.DOTALL)
        pattern_a = re.compile(r"<answer_start>(.*?)<answer_end>", re.DOTALL)
        question_match = re.search(pattern_q, qa_content)
        answer_match = re.search(pattern_a, qa_content)
        question = ''
        answer = ''
        
        if question_match and answer_match:
            question += question_match.group(1)
            answer += answer_match.group(1)
        else:
            continue

        print(question)
        print(answer)

        data = {'question': question, 'answer': answer}

        datas.append(data)

    with open('./test_qa_data.json', 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)


    