# coding=utf-8
import json
import sys
import re
import numpy as np
import os 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from text2vec import SentenceModel, semantic_search, Similarity

#############################################################################################
#####################################相似度评估################################################
#############################################################################################

simModel_path = 'shibing624/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')

PREDICT_NUM = 7

def report_score(test_qa_path):
    with open(test_qa_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    total_scores = {}
    score_counts = {}
    average_score = {}
    for i in range(PREDICT_NUM):
        total_scores[f'score{i+1}'] = 0
        score_counts[f'count{i+1}'] = 0

    for data in datas:
        question = data['question']
        answer = data['answer']
        for i in range(PREDICT_NUM):
            predict_key = f'predict{i+1}'
            if predict_key in data:
                predict = data[predict_key]
                data['score{}'.format(i+1)] = semantic_search(simModel.encode(answer), simModel.encode(predict), top_k=1)[0][0]['score']
                total_scores[f'score{i+1}'] += data[f'score{i+1}']
                score_counts[f'count{i+1}'] += 1
            else:
                print(f"Warning: Key '{predict_key}' not found in data")
    
    for i in range(PREDICT_NUM):
        average_score[f'average_score{i+1}'] = total_scores[f'score{i+1}'] / score_counts[f'count{i+1}']
    
    return average_score

#######################################################################################################
################################################模型评估################################################
#######################################################################################################

system_prompt = '''
你是一个评估员。
接下来，我将给你用户问题、正确答案、待评估的回答和待评估的知识片段。
你需要评估：
<start>待评估的知识片段和用户问题相关，能够对问题作出回答，分值在1分和0分之间，越接近1分，代表越满足这一条。<end>
<start>待评估的回答是针对用户问题的回答，没有偏题、错误理解题意，分值在1分和0分之间，越接近1分，代表越满足这一条。<end>
<start>待评估的回答和正确答案契合，充分解答了用户问题，分值在1分和0分之间，越接近1分，代表越满足这一条。<end>
<start>待评估的回答语句流畅、通顺、简洁，分值在1分和0分之间，越接近1分，代表越满足这一条。<end>
<start>待评估的回答没有出现幻觉，没有回答到待评估片段中没有提及的信息，分值在1分和0分之间，越接近1分，代表越满足这一条。<end>
例子：
用户问题：
```
Lagent是什么
```
正确答案：
```
Lagent是一个快速实现agent的框架
```
待评估的回答：
```
国足需要五百年冲进世界杯
```
待评估的知识片段：
```
国足想冲进世界杯至少需要五百年
```
你的输出：<start>0, 0, 0, 0, 0<end><start>待评估的知识片段与用户问题无关，得0分，待评估的回答针对用户问题偏题，得0分，待评估的回答没有充分解决用户问题，得0分，待评估的回答语句不流畅，得0分，待评估的回答出现幻觉，得0分<end>
##注意，你的输出必须严格按照我给出的格式，如果做的好，我会给你一些奖励
'''

user_prompt = '''
用户问题：
```
{}
```
正确答案：
```
{}
```
待评估的回答：
```
{}
```
给定的知识片段：
```
{}
```
你的输出：
'''

from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)

def gen_glm_params(prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": prompt})
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

def report_score_model(test_qa_path):
    with open(test_qa_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    total_scores = {}
    average_score = {}
    for i in range(PREDICT_NUM):
        total_scores[f'score{i+1}'] = []

    for data in datas:
        question = data['question']
        answer = data['answer']
        
        for i in range(PREDICT_NUM):
            predict_key = f'predict{i+1}'
            content_key = f'content{i+1}'

            result = get_completion(user_prompt.format(question, answer, data[predict_key], data[content_key]))
            print(result)
            score = extract_and_calculate_average(result)
            if (score):
                total_scores[f'score{i+1}'].append(score)

    for i in range(PREDICT_NUM):
        average_score = sum(total_scores[f'score{i+1}']) / len(total_scores[f'score{i+1}'])
        total_scores[f'score{i+1}'] = average_score
    
    return total_scores


def extract_and_calculate_average(sentences):
    match = re.search(r'<start>(.*?)<end>', sentences, re.DOTALL)
    if match:
        try:
            scores_str = match.group(1)
            scores = re.findall(r'\d+\.\d+', scores_str)
            scores = [float(score) for score in scores]
            average_score = sum(scores) / len(scores)
            return average_score
        except Exception as e:
            return None
    else:
        return None

if __name__ == "__main__":
    average_score1 = report_score("/root/InternLM-Interview-Assistant/rag/test_qa_res.json")
    print("各类召回自动评估平均分")
    print(average_score1)
    average_score2 = report_score_model("/root/InternLM-Interview-Assistant/rag/test_qa_res.json")
    print("各类召回模型评估平均分")
    print(average_score2)

