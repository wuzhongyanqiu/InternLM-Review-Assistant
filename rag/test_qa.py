import json
import sys
sys.path.append('/root/InternLM-Interview-Assistant/')

from server.tools.tools_worker import AnswerEvaluationTool

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/InternLM-Interview-Assistant", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./models/InternLM-Interview-Assistant", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

system_prompt = """
根据给出的上下文回答问题，你的答案要求简洁明了
给出的上下文：\n{}\n
问题：\n{}\n
##注意，如果无法从上下文中得到答案，说 "无答案"
你的回答：
"""

answerevaluationtool = AnswerEvaluationTool()

with open('/root/Mock-Interviewer/rag/test_qa_data.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)

for data in datas:
    question = data['question'].strip()
    rerank_ans1, rerank_ans2, emb_ans1, emb_ans2, emb_ans3, emb_ans4, bm25_ans = answerevaluationtool.test_rag(query=question)
    data['content1'] = rerank_ans1
    data['content2'] = rerank_ans2
    data['content3'] = emb_ans1
    data['content4'] = emb_ans2
    data['content5'] = emb_ans3
    data['content6'] = emb_ans4
    data['content7'] = bm25_ans
    prompt1 = system_prompt.format(rerank_ans1, question)
    prompt2 = system_prompt.format(rerank_ans2, question)
    prompt3 = system_prompt.format(emb_ans1, question)
    prompt4 = system_prompt.format(emb_ans2, question)
    prompt5 = system_prompt.format(emb_ans3, question)
    prompt6 = system_prompt.format(emb_ans4, question)
    prompt7 = system_prompt.format(bm25_ans, question)
    predict1, _ = model.chat(tokenizer, prompt1, history=[])
    predict2, _ = model.chat(tokenizer, prompt2, history=[])
    predict3, _ = model.chat(tokenizer, prompt3, history=[])
    predict4, _ = model.chat(tokenizer, prompt4, history=[])
    predict5, _ = model.chat(tokenizer, prompt5, history=[])
    predict6, _ = model.chat(tokenizer, prompt6, history=[])
    predict7, _ = model.chat(tokenizer, prompt7, history=[])
    data['predict1'] = predict1
    data['predict2'] = predict2
    data['predict3'] = predict3
    data['predict4'] = predict4
    data['predict5'] = predict5
    data['predict6'] = predict6
    data['predict7'] = predict7

with open('/root/Mock-Interviewer/rag/test_qa_res.json', 'w', encoding='utf-8') as f:
    json.dump(datas, f, ensure_ascii=False, indent=2)

    
    
