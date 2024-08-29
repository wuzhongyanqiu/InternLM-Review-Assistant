import sys
sys.path.append("/root/Mock-Interviewer")
from openai import OpenAI
import os
import erniebot
import json
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from web.api import get_answerevaluation
from server.tools.tools_prompt import interview_prompt_system, interview_prompt_input
from server.internvl.internvl_server import upload_pdf

##################################################################################
#########################Chat-GLM#################################################
##################################################################################

ZHIPUAI_API_KEY = os.environ['ZHIPUAI_API_KEY']
client = ZhipuAI(
    api_key=ZHIPUAI_API_KEY
)
def get_completion_glm(messages, model="glm-4", temperature=0.95):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

##################################################################################
#########################通义千问##################################################
##################################################################################

def get_response(messages):
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.environ['QWEN_KEY'], 
        # 填写DashScope服务的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
        temperature=0.95,
        top_p=0.8,
        )
    return completion


##################################################################################
#########################文心一言##################################################
##################################################################################

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ['ERNIE_KEY']

def get_completion_wenxin(messages, model="ernie-3.5", temperature=0.95, system_prompt_ernie=""):
    chat_comp = erniebot.ChatCompletion()

    resp = chat_comp.create(messages=messages, 
                        model=model,
                        temperature = temperature,
                        system=system_prompt_ernie,
                        )

    return resp["result"]

##################################################################################
#########################书生浦语##################################################
##################################################################################

def get_completion_puyu(messages):
    client = OpenAI(
        api_key=os.environ['INTERNLM_API_KEY'],  # 此处传token，不带Bearer
        base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
    )

    chat_rsp = client.chat.completions.create(
        model="internlm2.5-latest",
        messages=[{"role": "user", "content": "hello"}],
    )

    return chat_rsp.choices[0].message.content

##################################################################################
#########################生成多轮对话###############################################
##################################################################################

MAX_RESUMENUM = 200

prompt_gen_resume = """
- Role: 简历生成专家
- Background: 用户需要为不同行业生成虚拟简历，这些简历需要包含丰富的专业知识和详尽的内容，以满足不同职业角色的需求。
- Profile: 作为简历生成专家，你拥有广泛的行业知识和专业技能，能够根据不同的职业要求，创造性地构建简历内容。
- Skills: 你具备深厚的行业理解、写作能力、创造力和细节关注力，能够准确把握每个行业的专业术语和简历格式。
- Goals: 生成针对不同行业的简历，每个简历都具有高度的专业性和个性化特征。
- Constrains: 简历内容必须符合行业标准，避免使用虚假或误导性信息，同时保持创新性和吸引力。
- OutputFormat: 简历应包含个人信息、教育背景、工作经验、专业技能、项目经验、荣誉奖项、个人陈述等部分。
- Workflow:
  1. 确定简历所属的行业和职位。
  2. 收集该行业和职位所需的专业知识和技能。
  3. 创造性地构建简历内容，包括个性化的工作经历和专业技能描述。
  4. 确保简历格式规范，内容条理清晰，语言专业。
  5. 只输出简历内容，不要说其他多余的话
- Examples:
  - 例子1：金融行业投资分析师
    - 教育背景：经济学硕士，XX大学
    - 工作经验：在XX投资银行担任分析师，负责市场研究和投资策略制定
    - 专业技能：精通财务分析、风险评估、投资组合管理
  - 例子2：IT行业软件开发工程师
    - 教育背景：计算机科学学士，XX理工大学
    - 工作经验：在XX科技公司担任软件开发工程师，参与多个大型项目的开发
    - 专业技能：熟练掌握Java、Python编程语言，熟悉软件开发生命周期
  - 例子3：医疗行业临床研究员
    - 教育背景：医学博士，XX医学院
    - 工作经验：在XX医院担任临床研究员，参与多项临床试验
    - 专业技能：熟悉临床研究流程，具备数据分析和生物统计能力
"""

prompt_gen_job = """
- Role: 行业和职位接龙专家
- Background: 用户需要一个系统来生成不重复的行业和职位组合，以便于进行各种模拟、教育或娱乐活动。
- Profile: 你是一位对各行各业有深入了解的专家，能够创造性地生成多样化的行业和职位组合。
- Skills: 你具备广泛的知识储备、创新思维和记忆力，能够确保生成的组合既新颖又合理。
- Goals: 生成一系列不重复的行业和职位组合，满足用户的需求。
- Constrains: 确保每个生成的组合都是独一无二的，并且符合实际的职业设置。
- OutputFormat: 每个组合应简洁明了，格式统一，例如：“[行业] [职位]”。
- Workflow:
  1. 确定上一个生成的行业和职位组合。
  2. 思考并选择一个新的行业，确保不与之前的行业重复。
  3. 在选定的行业中创造性地选择或生成一个职位。
  4. 组合行业和职位，形成新的组合。
  5. 验证新组合是否符合实际，并确保不与之前的组合重复。
- Examples:
  - 例子1: 金融行业 投资分析师
  - 例子2: IT行业 软件开发工程师
  - 例子3: 医疗行业 临床研究员
  - 例子4: 教育行业 课程设计师
"""

prompt_interviewer = """
你是一个面试官，你能根据面试者的简历信息和对面试者进行面试，你一次只能提一个问题，你的问题必须与简历相关，涉及具体的专业知识，当面试者给出回答时，你可以进行点评和反问，你说话简洁明了，不超过300字。
"""

prompt_interviewer_post = """
你负责把这句话中的问题部分提取出来，只输出提取出来的句子，不要说其他无关的内容，比如
原句子为：<start>
面试官提问：
"你在之前的职位中，能否分享一个具体案例，说明你是如何通过制定个性化的营养支持方案，帮助运动员提高他们的竞技表现的？" 
（点评与反问将基于面试者的回答进行。）<end>
你的输出：
你在之前的职位中，能否分享一个具体案例，说明你是如何通过制定个性化的营养支持方案，帮助运动员提高他们的竞技表现的？
原句子为：<start>
{}
<end>
你的输出:
"""

prompt_candidate = """
你是一个面试者，你将根据历史对话内容回答面试官的问题，作为一个初出茅庐的校招生，你的回答不能过于完美，同时要注重简洁明了，不超过300字。
"""

prompt_candidate_post = """
你负责把这段回答精炼简化成只有基本的逗号、句号标点的段落，要求简洁明了，保留原句子的语义，只输出精炼后的句子，不要说其他无关内容
原句子为：{}
你的输出：
"""

def gen_job():
    messages_a = [{'role': 'system', 'content': prompt_gen_job}]
    messages_b = [{'role': 'system', 'content': prompt_gen_job}]
    messages_a.append({'role': 'user', 'content': '法律行业 知识产权律师'})
    messages_b.append({'role': 'user', 'content': '法律行业 知识产权律师'})
    messages_a.append({'role': 'assistant', 'content': '制造业 工艺工程师'})
    messages_b.append({'role': 'assistant', 'content': '制造业 工艺工程师'})
    messages_a.append({'role': 'user', 'content': '媒体行业 新媒体运营专家'})
    for i in range(MAX_RESUMENUM):
        job_res_a = get_completion_glm(messages_a)
        yield job_res_a
        messages_a.append({'role': 'assistant', 'content': job_res_a})
        messages_b.append({'role': 'user', 'content': job_res_a})
        job_res_b = get_completion_glm(messages_b)
        yield job_res_b
        messages_a.append({'role': 'user', 'content': job_res_b})
        messages_b.append({'role': 'assistant', 'content': job_res_b})

def gen_resumes():
    job_generator = gen_job()
    while True:
        try:
            job_info = next(job_generator)
            resume = gen_resumes_for_job(job_info)
            yield resume
        except StopIteration:
            break

def gen_resumes_for_job(job_info):
    messages = [{'role': 'system', 'content': prompt_gen_resume}]
    messages.append({'role': 'user', 'content': job_info})
    resume_res = get_completion_glm(messages)
    return resume_res

def gen_multi_chat(resume):
    conversation = {}
    conversation['conversation'] = []
    messages_interviewer = [{'role': 'system', 'content': prompt_interviewer}]
    messages_candidate = [{'role': 'system', 'content': prompt_candidate}]
    messages_interviewer.append({'role': 'user', 'content': '面试官您好，我的简历是{}'.format(resume)})
    messages_candidate.append({'role': 'user', 'content': '我是你这次面试的面试官，请你先介绍一下自己的简历'})
    messages_candidate.append({'role': 'assistant', 'content': '好的，这是我的简历：\n{}'.format(resume)})

    ques = get_completion_glm(messages_interviewer)
    messages_interviewer_post = [{'role': 'user', 'content': prompt_interviewer_post.format(ques)}]
    ques_post = get_completion_glm(messages_interviewer_post)

    messages_candidate.append({'role': 'user', 'content': ques_post})
    messages_interviewer.append({'role': 'assistant', 'content': ques_post})
    tmp_dict_outside = {
        "system": prompt_interviewer,
        "input": '面试官您好，我的简历是{}'.format(resume),
        "output": ques_post
    }
    conversation['conversation'].append(tmp_dict_outside)
    
    for i in range(5):
        ans = get_completion_glm(messages_candidate)

        messages_candidate_post = [{'role': 'user', 'content': prompt_candidate_post.format(ans)}]
        ans_post = get_completion_glm(messages_candidate_post)

        messages_candidate.append({'role': 'assistant', 'content': ans_post})
        messages_interviewer.append({'role': 'user', 'content': ans_post})
 
        ques = get_completion_glm(messages_interviewer)
        messages_interviewer_post = [{'role': 'user', 'content': prompt_interviewer_post.format(ques)}]
        ques_post = get_completion_glm(messages_interviewer_post)

        messages_candidate.append({'role': 'user', 'content': ques_post})
        messages_interviewer.append({'role': 'assistant', 'content': ques_post})
        tmp_dict_inside = {
            "input": ans,
            "output": ques_post
        }
        conversation['conversation'].append(tmp_dict_inside)

    return conversation

def save_conversations(conversations, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

##################################################################################
#########################生成单轮对话###############################################
##################################################################################

chat_prompt1 = "用简单精炼的话回答这个问题，<start>{}<end>，不多于100字，不要说其他无关的内容。"

check_ques = "识别<start>{}<end>是不是一个无效问题，无效问题是指必须参考图片、表格或者书籍才能回答的问题，如果是，输出<start>无效问题<end>，无效问题示例：<start>这个图片中描述了什么<end><start>第三章讲了什么内容<end><start>请描述一下这个图片的特点和用途？<end>，注意，你只需要给出判断结果，不需要解释，你能够利用已有知识回答的问题都是有效问题"

import sqlite3

def view_db_contents(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM questions")

    questions = cursor.fetchall()

    cursor.close()
    conn.close()

    return questions

def gen_chat(questions):
    conversations = []
    i = 0
    for question in questions:
        question_check_messages = [{'role': 'user', 'content': check_ques.format(question)}]
        question_check = get_completion_glm(question_check_messages)
        print(question_check)
        if "无效问题" in question_check:
            print("这是一个无效问题-------------------------------------------")
            print(question)
            continue
        i = i + 1
        print("i={}".format(i))
        print(question)
        conversation = {}
        conversation['conversation'] = []
        messages1 = [{'role': 'user', 'content': chat_prompt1.format(question[1])}]
        ans = get_completion_glm(messages1)
        print("ans----------------------------------------------------")
        print(ans)
        print("q+a---------------------------------------------------")
        print(question[1]+ans)
        rag_ans = get_answerevaluation(question[1], ans, question[1]+ans)
        print("rag_ans------------------------------------------------")
        print(rag_ans)
        messages2 = [{'role': 'system', 'content': interview_prompt_system}]
        input_content = interview_prompt_input.format(rag_ans)
        messages2.append({'role': 'user', 'content': input_content})
        output_content = get_completion_glm(messages2)
        print("output_content---------------------------------------------")
        print(output_content)
        tmp_dict_inside = {
            "system": interview_prompt_system,
            "input": input_content,
            "output": output_content
        }
        conversation['conversation'].append(tmp_dict_inside)
        conversations.append(conversation)
        save_conversations(conversations, '/root/Mock-Interviewer/datas/chat_new1.json')

if __name__ == "__main__":
    # conversations = []
    # for resume in gen_resumes():
    #     conversation = gen_multi_chat(resume)
    #     conversations.append(conversation)

    #     # 每次迭代后保存对话到文件
    #     save_conversations(conversations, '/root/Mock-Interviewer/datas/multi_interview_new.json')

    # upload_pdf('/root/Mock-Interviewer/datas/knowledge.pdf')
    quesions = view_db_contents('/root/Mock-Interviewer/storage/db_questions.db')
    gen_chat(quesions)

    