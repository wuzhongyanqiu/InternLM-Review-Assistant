import csv
import json
import re
from zhipuai import ZhipuAI

ZHIPUAI_API_KEY = "YOUR API KEY"
client = ZhipuAI(
    api_key=ZHIPUAI_API_KEY
)

def chunk_knowledge(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 使用正则表达式匹配所有二级标题
        headers = re.findall(r'(## .*)', content)
        # 以二级标题分割内容
        chunks = re.split(r'(## .*)', content)    
    # 移除所有包含 '#' 的元素
    chunks = [chunk for chunk in chunks if not chunk.startswith(('# ', '##', '#'))]
    return chunks


def generate_question(content):
    system_prompt = """
    你是一个面试官，你擅长将现有知识点改写成一个面试题来询问面试者。\n
    根据以下提供的现有知识点：\n\n#############\n{}#############\n 改写成一个面试题\n
    要求，改写的面试题是和现有知识点相关的问题。\n
    要求，你只输出你的面试题，你的面试题仅是一个包含基本逗号、句号的段落，不要包含其他字符或其他结构。 \n
    """.format(content)
    return system_prompt

def generate_wrong_answer(content):
    system_prompt = """
    你是一个面试者，你的基础知识并不是很牢固，当面试官询问你一道面试题时，你将给出一个并不完美的答案，其中会包含一些明显的错误。\n
    面试官提出的问题是: \n\n#############\n{}#############\n 给出这个问题的答案\n
    要求，给出的答案要体现出面试者的知识有所缺陷，包含明显的错误，但是大致思路是正确的\n
    要求，你的答案仅是一个包含基本逗号、句号的段落，不要包含其他字符或其他结构。 \n
    """.format(content)
    return system_prompt

def generate_evaluation_answer(question, content):
    system_prompt = """
    你是一个面试官，当面试者给出面试题的答案时，你会评估他的答案是否正确，是否有明显的错误，你将用认真严谨的态度给予点评，同时改正他的答案。\n
    面试题是：\n\n#############\n{}#############\n 
    面试者给出的的答案是: \n\n#############\n{}#############\n 给出你的点评\n
    要求，给出的点评要正确改正面试者答案里的错误，对面试者有帮助，点评语言需要严谨认真。\n
    要求，你的点评仅是一个包含基本逗号、句号的段落，不要包含其他字符或其他结构。 \n
    """.format(question, content)
    return system_prompt

def generate_conversation(question, wrong_ans, right_ans):
    conversation = {}
    conversation['conversation'] = {
        "system": "你是一个面试官，当面试者给出面试题的答案时，你会评估他的答案是否正确，是否有明显的错误，你将用认真严谨的态度给予点评，同时改正他的答案。",
        "input": "面试题是：{}, 面试者给出的的答案是: {}".format(question, wrong_ans), 
        "output": right_ans
    }
    return conversation

def gen_glm_params(prompt):
    messages = [{"role": "user", "content": prompt}]
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

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

if __name__ == "__main__":
    FILE_NUMS = 11
    conversations = []
    conversation = {}
    # input_question = ""
    # input_answer = ""
    # output = ""

    for file_num in range(FILE_NUMS):
        md_file = '../datas/{}.md'
        knowledges = chunk_knowledge(md_file.format(file_num+1))
        for knowledge in knowledges:
            question_prompt = generate_question(knowledge)
            # print("问题提示为：\n" + question_prompt)
            input_question = get_completion(prompt=question_prompt)
            # input_question = res
            # 使用正则表达式匹配花括号内的内容
            # res = re.findall(r"```python(.*)```", input_question, flags=re.DOTALL)[0].strip()
            # res = res.replace("'", '"')
            # try:
            #     input_question = json.loads(res)['改写的面试题']
            # except:
            #     continue
            wrong_answer_prompt = generate_wrong_answer(content=input_question)
            # print("错误答案提示为：\n" + wrong_answer_prompt)
            input_answer = get_completion(wrong_answer_prompt)

            # print(input_answer)

            evaluation_prompt = generate_evaluation_answer(question=input_question, content=input_answer)
            # print("答案评估提示为：\n" + evaluation_prompt)
            output = get_completion(evaluation_prompt)
            # print(output)

            conversation = generate_conversation(input_question, input_answer, output)
            print(conversation)
            conversations.append(conversation)
 

            # print(result)
            # json_pattern = r'\{.*?\}'
            # json_strings = re.findall(json_pattern, result, re.DOTALL)
            # for json_str in json_strings:
            #     conversations.append(json_str)

    with open('../datas/interview_data.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
      
