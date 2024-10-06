import requests
import json
import re
import os
import sqlite3
from lmdeploy.serve.openai.api_client import APIClient
import threading
import concurrent.futures
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))

DIR_PATH = os.path.join(current_dir, "../tmp_dir/datas")
MINERU_OUTPUTDIR = os.path.join(current_dir, "MinerU_output")
RESUME_PATH = os.path.join(current_dir, "../tmp_dir/resume/resume.pdf")
DB_PATH = os.path.join(current_dir, "storage/database")

def deepseek_response(messages):
    from openai import OpenAI

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )

    return response.choices[0].message.content

def local_response(messages):
    from openai import OpenAI

    client = OpenAI(
        api_key = "internlm2",
        base_url = "http://0.0.0.0:23333/v1"
    )

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages
    )

    return response.choices[0].message.content

def save_conversations(conversations, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

def get_response(messages):
    from openai import OpenAI

    client = OpenAI(api_key="sk-abfghsaljbqtabrkjeexxwegooutwyovhsgywhzrjbxdotws", base_url="https://api.siliconflow.cn/v1")

    response = client.chat.completions.create(
        model='internlm/internlm2_5-7b-chat',
        # internlm/internlm2_5-20b-chat
        # Qwen/Qwen2-72B-Instruct
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content

# 用于线程安全的文件写入
file_lock = threading.Lock()
batch_size = 32

MESSAGE2_PROMPT = '根据你已有的知识和上下文内容，回答问题，要求语言简洁通顺，答案准确无误，注意，当上下文内容与问题无关时，不要编造答案，输出无答案'
MESSAGE3_PROMPT = '根据你已有的知识和参考答案，对面试者的答案进行点评，你直接与面试者对话，因此要符合对话的语法，如果面试者回答过于简单或者错误百出，你会严格的进行批评和指正'
file_path = 'storage/single_chat.json'

class QAComment:
    # 初始化对话列表
    conversations = []
    
    def get_contentdb(self):
        url = "http://0.0.0.0:8004/rag/gen_contentdb"
        requests.post(url)

    def get_questiondb(self):
        url = "http://0.0.0.0:8004/rag/gen_questiondb"
        requests.post(url)

    def fetch_questions_from_db(self, db_filename=DB_PATH):
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        
        cursor.execute('SELECT question FROM questions')
        rows = cursor.fetchall()
        
        questions = []
        for row in rows:
            questions.append(row[0])  # 取出并打印每个问题
        
        conn.close()
        return questions

    # 定义多线程处理函数
    def process_query(self, query, MESSAGE2_PROMPT, MESSAGE3_PROMPT, file_path):
        conversation1 = {"conversation": []}
        conversation2 = {"conversation": []}

        # 生成面试问题
        messages = [{'role': 'user', 'content': f'改写这句话：{query}，使其成为一个严谨的面试题，要求简短，仅输出面试题本身，不要输出其他无关内容'}]

        question = deepseek_response(messages)

        # 生成面试者的错误回答
        system_prompt1 = '你要扮演一个基础很差的面试者，在回答面试问题的时候你会犯很多错误，注意，你一定要说出很多错误，你只回答问题，不要输出其他无关内容'
        messages1 = [{'role': 'system', 'content': system_prompt1}, {'role': 'user', 'content': question}]

        ans = deepseek_response(messages1)

        # 调用外部 API 获取评论内容
        url = "http://0.0.0.0:8004/rag/quicklyQA_comments"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"quicklyQA_query": question, "quicklyQA_ans": ans})
        response = requests.post(url, data=payload, headers=headers).json()
        content_text, rightans_text, comments_text = response['content'], response['rightans'], response['comments']

        conversation1['conversation'].append({"system": "", "input": query, "output": content_text})

        conversation2['conversation'].append({"system": "", "input": ans, "output": comments_text})

        return [conversation1, conversation2]

    # 多线程执行
    def process_queries_in_parallel(self, MESSAGE2_PROMPT, MESSAGE3_PROMPT, file_path):
        queries = self.fetch_questions_from_db()
        results = []  # 用于存储每个线程的结果
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_query, query, MESSAGE2_PROMPT, MESSAGE3_PROMPT, file_path) for query in queries]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()  # 获取每个线程的结果
                print(result)
                if result:  # 如果结果有效，添加到结果列表
                    results.append(result[0])
                    results.append(result[1])
                    # 每处理完一批线程就写入一次
                if (i + 1) % batch_size == 0:
                    with file_lock:
                        self.conversations = self.conversations + results
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(self.conversations, f, ensure_ascii=False, indent=4)
                    results.clear()  # 清空结果列表以准备下一批

        # 处理剩余结果
        if results:
            with file_lock:
                self.conversations = self.conversations + results
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.conversations, f, ensure_ascii=False, indent=4)
            results.clear()  # 清空结果列表以准备下一批


TOOL_DESCRIPTION = """
system name=<|plugin|>
你可以使用如下工具：
[
    {
        "name": "MockInterview.get_comments",
        "description": "This is the subfunction for tool 'MockInterview', you can use this tool. The description of this function is: \n在模拟面试期间，当面试者刚刚回答了一个专业的技术问题，面试官想对其回答的深度和准确性进行评价时，可以使用这个 API 得到参考信息。\n   在模拟面试期间，当面试者提供了很好的解决方案，面试官希望给予正面的反馈并建议一些提升点时。可以使用这个 API 得到需要的参考信息。",
        "parameters": [
            {
                "name": "mockinterview_query",
                "type": "STRING",
                "description": "问题"
            },
            {
                "name": "mockinterview_ans",
                "type": "STRING",
                "description": "面试者的回答"
            }
        ],
        "required": [
            "mockinterview_query",
            "mockinterview_ans"
        ],
        "parameter_description": "If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name."
    },
    {
        "name": "MockInterview.get_questions",
        "description": "This is the subfunction for tool 'MockInterview', you can use this tool. The description of this function is: \n在模拟面试期间，当面试者对某个技术领域的回答不够深入，面试官决定根据对话内容选择一些逗号分隔的关键词作为参数，从题库中中抽取一些相关的问题，以进一步评估面试者的知识时，可以使用这个 API 得到需要的相关问题。\n   在模拟面试期间，当一个话题结束，面试官希望根据技术栈出一道专业问题来对面试者提问时，可以使用这个 API 得到需要的相关问题。",
        "parameters": [
            {
                "name": "mockinterview_keywords",
                "type": "STRING",
                "description": "要进行题库查找的关键词，用逗号分隔"
            }
        ],
        "required": [
            "mockinterview_keywords"
        ],
        "parameter_description": "If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name."
    },
    {
        "name": "MockInterview.get_resumes",
        "description": "This is the subfunction for tool 'MockInterview', you can use this tool. The description of this function is: \n在模拟面试期间，面试官发现有提到的项目或技能需要进一步探讨，可以获取简历了解面试者项目或技能的细节以便提问时，可以使用这个 API 得到简历信息。\n   在模拟面试期间，在面试过程中，面试官希望得知面试者的技术栈时，可以使用这个 API 得到简历信息。",
        "parameters": [],
        "required": [],
        "parameter_description": "If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name."
    },
    {
        "name": "QuicklyQA.get_query",
        "description": "This is the subfunction for tool 'QuicklyQA', you can use this tool. The description of this function is: \n在快问快答期间，当轮到你提问时，使用这个 API 从题库中抽取一道题提问。\n   在快问快答期间，当用户回答完问题，而且你对其回答进行评估以后，使用这个 API 再次进行提问。",
        "parameters": [],
        "required": [],
        "parameter_description": "If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name."
    },
    {
        "name": "QuicklyQA.get_comments",
        "description": "This is the subfunction for tool 'QuicklyQA', you can use this tool. The description of this function is: \n在快问快答期间，当用户回答了你提出的问题后，使用这个 API 得到评估的信息。",
        "parameters": [
            {
                "name": "quicklyQA_query",
                "type": "STRING",
                "description": "问题"
            },
            {
                "name": "quicklyQA_ans",
                "type": "STRING",
                "description": "面试者的回答"
            }
        ],
        "required": [
            "quicklyQA_querymum",
            "quicklyQA_ans"
        ],
        "parameter_description": "If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name."
    }
]
如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! 
同时注意你可以使用的工具，不要随意捏造！
"""

SYSTEM_PROMPT = f'''
当开启工具以及代码时，根据需求选择合适的工具进行调用。
{TOOL_DESCRIPTION}
'''

### 统一用智能体的systemprompt，然后第一个对话是开始模拟面试，或者开始快问快答，快问快答的回答是我将调用什么工具，之后得到工具结果之后进行提问，模拟面试也一样###


class Getdatas:
    messages = []
    conversations = []

    def init_messages(self):
        self.messages = []
        self.messages.append({'role': 'system', 'content': SYSTEM_PROMPT})

    def convert_to_finetuneform(self, messages):
        conversation = {"conversation": []}
        if messages[0]['role'] == 'system':
            conversation['conversation'].append({'system': messages[0]['content'], 'input': messages[1]['content'], 'output': messages[2]['content']})
            messages = messages[3:]
        
        for i in range(0, len(messages), 2):
            conversation['conversation'].append({'input': messages[i]['content'], 'output': messages[i+1]['content']})

        tmp_conversation = deepcopy(conversation)

        self.conversations.append(tmp_conversation)

    def save_json(self, save_path):
        save_conversations(self.conversations, 'storage/multi_chat.json')
    
    def get_mockinterview_messages(self):
        self.init_messages()
        self.messages.append({'role': 'user', 'content': '你好！'})
        self.messages.append({'role': 'assistant', 'content': '你好！有什么我可以帮助你的吗？你可以直接对我提问。我也可以作为你的复习伙伴，为你提供模拟面试或者快问快答服务。'})
        self.messages.append({'role': 'user', 'content': '开始模拟面试'})
        self.messages.append({'role': 'assistant', 'content': '好的，接下来将开始模拟面试，请上传你的简历'})
        self.messages.append({'role': 'user', 'content': '我已经上传简历'})
        self.messages.append({'role': 'assistant', 'content': '''好的，我将使用MockInterview工具中的get_resumes函数来查看你的简历内容，请稍等片刻<|action_start|><|plugin|>
{"name": "MockInterview.get_resumes", "parameters": {}}<|action_end|>'''})
        resume_messages = [{'role': 'user', 'content': '根据给定格式，虚构一个内容丰富的简历，要求多样性，格式为：\n项目经历：\n实习经历：\n科研经历：\n技术栈：\n, 例子：项目经历:\n1. 面向微服务架构的分布式日志系统\n时间: 2020年9月 - 2021年1月\n项目描述：设计并开发了一个分布式日志系统，用于微服务架构下的日志收集、聚合与实时分析。系统使用ELK（Elasticsearch, Logstash, Kibana）技术栈实现日志的集中管理，并结合Kafka提高日志传输的实时性与可靠性。\n职责：\n使用Elasticsearch搭建分布式日志存储系统，确保数据的高可用性。\n集成Kafka和Logstash，完成日志数据的实时传输和处理。\n在Kibana上进行日志可视化展示，方便系统运维和监控。\n技术栈：Elasticsearch、Kafka、Logstash、Kibana、Docker、Microservices\n\n2. 基于Java的多线程电子商务平台\n时间: 2019年10月 - 2020年3月\n项目描述：开发一个高性能的电子商务平台，支持多用户的同时操作。平台采用Java多线程技术优化响应速度，结合Redis进行缓存，加快订单处理流程，提升用户体验。\n职责：\n使用Java设计并实现多线程订单处理系统。\n集成Redis进行缓存管理，优化数据查询速度。\n通过JMeter进行性能测试，并分析优化系统瓶颈。\n技术栈：Java、Spring Boot、MySQL、Redis、JMeter\n\n3. 深入浅出Python数据分析与可视化平台\n时间: 2021年2月 - 2021年6月\n项目描述：搭建一个数据分析与可视化平台，能够根据用户输入的数据进行分析并生成动态可视化报表，广泛应用于市场分析、运营数据监控等领域。\n职责：\n使用Pandas和NumPy进行数据处理和分析。\n使用Matplotlib和Seaborn库进行可视化图表的生成。\n构建Django框架下的Web平台，支持用户上传数据并查看报表。\n技术栈：Python、Pandas、NumPy、Matplotlib、Seaborn、Django\n\n科研经历:\n1. 基于机器学习的医疗文本分类研究\n时间: 2020年3月 - 2020年12月\n描述：研究如何使用机器学习技术对医疗领域的文本数据进行分类，探索不同算法如朴素贝叶斯、随机森林、XGBoost在医疗文本分类任务中的表现。通过特征工程和模型优化，提升分类的准确率。\n职责：\n使用TF-IDF对医疗文本进行特征提取。\n采用随机森林和XGBoost进行文本分类，并对比不同模型的性能。\n使用交叉验证优化模型，并撰写科研报告。\n技术栈：Python、scikit-learn、XGBoost、NLTK、Pandas\n\n2. 人工智能在智能推荐系统中的应用\n时间: 2019年10月 - 2020年6月\n描述：研究人工智能算法在智能推荐系统中的应用，重点探索协同过滤算法与基于内容的推荐算法的结合，提升推荐系统的精度。最终提出了一种混合推荐策略，在用户行为数据上取得了较好的效果。\n职责：\n设计并实现基于协同过滤和内容推荐的算法。\n使用Python构建数据预处理管道，并进行模型训练与测试。\n进行A/B测试验证推荐系统效果，并优化算法策略。\n技术栈：Python、scikit-learn、Surprise库、Pandas、NumPy\n\n实习经历:\n美团 - 后端开发实习生\n时间: 2021年6月 - 2021年9月\n职责：\n参与美团外卖平台的后端开发，优化配送系统的路径规划算法。\n使用Java和Spring Boot优化平台订单处理流程，提高系统响应速度。\n参与开发Redis缓存机制，减少数据库查询压力。\n技术栈：Java、Spring Boot、Redis、MySQL、RabbitMQ\n\n滴滴出行 - 数据工程实习生\n时间: 2020年7月 - 2020年9月\n职责：\n负责处理滴滴平台上海量的出行数据，进行数据清洗与分析。\n使用Hadoop和Spark进行大规模数据处理，并生成分析报表。\n参与实时数据管道的构建，优化数据处理效率。\n技术栈：Hadoop、Spark、Hive、Python、SQL\n\n技术栈:\n编程语言: Java, Python, Go, JavaScript, SQL\n框架与库: Spring Boot, Django, Flask, TensorFlow, XGBoost\n数据库: MySQL, PostgreSQL, Redis, Elasticsearch\n工具: Docker, Kubernetes, Git, Hadoop, Kafka, Spark\n开发技能: 分布式系统设计, 微服务架构, 高并发处理, 大数据处理, 数据分析与建模\n算法: 机器学习, 深度学习, 自然语言处理 (NLP), 协同过滤, 图神经网络\n语言\n中文（母语）\n英文（流利）'}]
        resume_content = deepseek_response(resume_messages)
        print(resume_content)
        self.messages.append({'role': 'user', 'content': f'environment name=<|plugin|>{resume_content}'})

        subtext1 = '''
        1. 你的简历很有趣，我们开始面试吧。首先，根据你的项目经历，我想询问在 InternLM-Interview-Assistant 项目中，你是如何利用 RAG 构建知识库和面试题库的？
        2. 你的简历写的不错，我看到你这里有一个叫作智能座舱助手的项目，你能讲讲你这个项目具体做了什么吗，你在项目中扮演了什么角色？
        3. 我注意到你在 InternLM-Interview-Assistant 项目中，对知识库构建与检索器的创建有深入研究，特别是在处理不同类型的 PDF 文件上，你能说说不同类型具体指什么吗，分别是怎么处理的？
        4. 你简历上提到了两个项目，那么你先选一个你最感兴趣的介绍一下吧
        5. 我看了一下你的简历，你在小米集团实习过三个月，你能说说这段经历给你的收获吗，还有你在其中学到了什么？
        '''

        subtext1_user = f'这里有一份简历，内容是:\n{resume_content}\n，根据这个简历写一个类似于示例的句子，示例为:\n{subtext1}\n，只输出这个句子，不要说其他无关内容，你仅输出一个句子，要求多样性'
        
        submessages1 = [{'role': 'user', 'content': subtext1_user}]

        subtext1_assistant = deepseek_response(submessages1)
        self.messages.append({'role': 'assistant', 'content': subtext1_assistant})

        subtext2_user = f'这是你的简历:\n{resume_content}\n，回答这个问题:\n{subtext1_assistant}\n，你只输出你的回答，你的回答仅包含简单的逗号句号，不要说其他无关的内容'
        submessages2 = [{'role': 'user', 'content': subtext2_user}]
        subtext2_assistant = deepseek_response(submessages2)

        self.messages.append({'role': 'user', 'content': subtext2_assistant})

        subtext3_action1 = '<|action_start|><|plugin|>{"name": "MockInterview.get_questions", "parameters": {"mockinterview_keywords": "非牛顿流体,格子玻尔兹曼"}}<|action_end|>'
        subtext3_action2 = '<|action_start|><|plugin|>{"name": "MockInterview.get_questions", "parameters": {"mockinterview_keywords": "OCR,检索器"}}<|action_end|>'
        subtext3_user = f'''
        对话上下文是：
        看到你在科研经历中提到了一篇关于非牛顿流体的论文，能否详细分享一下你是如何运用低存储伴随格子玻尔兹曼方法来控制非牛顿流体流动，特别是在组织流中估计应力分布的应用过程？
        在项目中,我首先深入研究了低存储伴随格子玻尔兹曼方法的基础理论,理解其在流体力学模拟中的优势,特别是在处理高维,非线性问题时的高效性,接着,针对非牛顿流体的特点,调整了格子玻尔兹曼碰撞算子,使其能够准确描述剪切变稀或剪切变稠的流体行为,然后,设计并实现了低存储策略,显著减少了计算资源的需求,特别是在处理大规模流场模拟时,这一步是通过优化数据结构和算法,减少不必要的存储和计算,实现了高效求解伴随方程,在应用到组织流中估计应力分布时,我们首先建立了一个详细的生物流体力学模型,涵盖了组织的复杂几何和非均匀性质,然后,运用低存储伴随格子玻尔兹曼方法,对模型进行精细化模拟,计算出在不同流速,剪切力条件下组织内部的应力分布,这为理解生物力学效应,如细胞损伤,提供了关键信息,最后,通过对比实验数据,验证了我们模型的准确性和可靠性,为未来进一步研究打下了坚实基础,在这一过程中,不仅提升了我的科研技能,也增强了我解决实际问题的能力.

        回答是：
        你的回答中提到了非牛顿流体和格子玻尔兹曼模型，我将抽取一些相关的问题来问你{subtext3_action1}

        对话上下文是：
        我注意到你在 InternLM-Interview-Assistant 项目中，对知识库构建与检索器的创建有深入研究，特别是在处理不同类型的 PDF 文件上，你能说说不同类型具体指什么吗，分别是怎么处理的？
        我利用 minerU 来处理 PDF 文件，minerU 是一个进行 PDF 解析的开源工具，其首先先根据 PDF 布局和内容来将其分类为正常 PDF 和需要 OCR 识别的 PDF，然后对其进行不同的处理，在此之后，我得到了一个识别后的文本集，公式和表格以 markdown 格式嵌入在了文本里，之后进行切分和检索器构建。

        回答是：
        你提到的 minerU 是利用 OCR 进行识别的，在之后进行检索器的构建，我将抽取一些相关的问题来问你{subtext3_action2}
        
        对话上下文是：
        {subtext1_assistant}
        {subtext2_assistant}

        回答是：
        '''

        submessages3 = [{'role': 'user', 'content': subtext3_user}]
        subtext3_assistant = deepseek_response(submessages3)
        self.messages.append({'role': 'user', 'content': subtext3_assistant})

        # 使用正则表达式提取 mockinterview_keywords 对应的值
        pattern = r'"mockinterview_keywords":\s*"([^"]*)"'
        match = re.search(pattern, subtext3_assistant)
        mockinterview_keywords = match.group(1)

        print(mockinterview_keywords)

        url = "http://0.0.0.0:8004/rag/mockinterview_questions"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"mockinterview_keywords": mockinterview_keywords})

        questions = requests.post(url, data=payload, headers=headers).json()

        questions_content = questions['question']
        
        self.messages.append({'role': 'user', 'content': f'environment name=<|plugin|>{questions_content}'})

        subtext4_action1 = '<|action_start|><|plugin|>{"name": "MockInterview.get_questions", "parameters": {"mockinterview_keywords": "C++, CUDA, 非牛顿流体, 低存储, 计算加速"}}<|action_end|>'
        subtext4_action2 = '<|action_start|><|plugin|>{"name": "MockInterview.get_questions", "parameters": {"mockinterview_keywords": "数据存储结构优化,缓存机制,负载均衡,网络通信协议优化,服务器集群管理"}}<|action_end|>'
        subtext4_user = f'''
        对话上下文是：
        你提到使用了C++和CUDA来优化计算过程，特别是在处理非牛顿流体控制问题时，我将抽取一些相关的问题来问你{subtext4_action1}
        environment name=<|plugin|>关于C++的面试题包括: \n['1. 如何在C++中使用抽象工厂模式？', '2. 在 C++ 中，如何利用私有头文件隐藏类的实现部分？']\n关于 CUDA的面试题包括: \n['4. CUDA 8.0的安装过程中，用户被提供了哪些选择？', '裁剪后图像保留主要物体，KSH论文核心思想概述，忽略细节可参考论文，Alex Krizhevsky cuda-convnet实现，Theano 实现代码，多GPU使用复杂性，Caffe框架KSH网络版本，Model Zoo参考。']\n关于 非牛顿流体的面试题包括: \n[]\n关于 低存储的面试题包括: \n[]\n关于 计算加速的面试题包括: \n[]

        回答是：
        从你提及的技能和背景中，我发现你对C++以及CUDA都很熟悉，接下来我将向你提问一道与C++相关的题目，在C++中，你如何使用抽象工厂模式？期待你的详细回答。

        对话上下文是：
        你提到了在MMORPG项目中优化服务器性能的多个方面，包括数据存储结构的优化、缓存机制的引入、负载均衡、网络通信协议的优化以及服务器集群的高效管理，我将抽取一些相关的问题来问你{subtext4_action2}
        environment name=<|plugin|>关于数据存储结构优化的面试题包括: \n[]\n关于缓存机制的面试题包括: \n['2. cookie与会话管理和缓存机制之间的关系是什么？', '9. 解释为什么InnoDB的缓存机制与MyISAM不同。']\n关于负载均衡的面试题包括: \n['5. 在微服务架构中，如何实现高可用和负载均衡？', '1. 负载均衡器是如何扩展服务实例清单的？']\n关于网络通信协议优化的面试题包括: \n[]\n关于服务器集群管理的面试题包括: \n[]
        
        回答是：
        尽管我没有找到完全切题的面试题，但是让我们尝试这道题：你能解释一下cookie与会话管理和缓存机制之间的关系吗？这会进一步探究你在应用缓存机制上的理解。

        对话上下文是：
        {subtext3_assistant}
        environment name=<|plugin|>{questions_content}

        回答是：
        '''

        subtextmessages4 = [{'role': 'user', 'content': subtext4_user}]
        subtext4_assistant = deepseek_response(subtextmessages4)
        self.messages.append({'role': 'assistant', 'content': subtext4_assistant})

        subtext5_user = f'回答这个问题:\n{subtext4_assistant}\n，你只输出你的回答，你的回答仅包含简单的逗号句号，不要说其他无关的内容'
        submessages5 = [{'role': 'user', 'content': subtext5_user}]
        subtext5_assistant = deepseek_response(submessages5)

        self.messages.append({'role': 'user', 'content': subtext5_assistant})
        
        subtext6_action1 = '<|action_start|><|plugin|>{"name": "MockInterview.get_comments", "parameters": {"mockinterview_query": "在你描述的"智能数据分析系统"中，数据清洗是非常关键的步骤。我想了解一下你如何利用STREAM操作进行数据清洗，特别是在去除重复数据或无关数据方面有什么经验或技巧？这将帮助我们更深入地了解你的实践能力。", "mockinterview_ans": "使用STREAM操作进行数据清洗时,优先评估数据流的特性,确定清洗策略,采用滑动窗口减少内存占用,实时监测数据质量,及时调整参数,去除重复数据,利用keyBy对数据进行分组,通过窗口函数比较相同键值的数据,保留最新或最常用记录,过滤无关数据,设置过滤条件,如数据类型,时间戳等,确保数据流的纯净和有效性,实践中,持续监控清洗效果,优化算法,提高数据处理效率和准确性."}}<|action_end|>'
        subtext6_action2 = '<|action_start|><|plugin|>{"name": "MockInterview.get_comments", "parameters": {"mockinterview_query": "从你提到的技术中，让我们先探讨关于数据增强的问题。在处理图像数据时，你如何进行数据增强，包括转置图像通道和限制像素值范围？这是理解你在深度学习中如何有效地使用数据增强技术的关键问题。期待你的详细解答。如果对于学习率策略也有相关的问题，你同样可以准备和分享，这样我们可以更全面地深入了解你在深度学习模型优化方面的知识和经验。", "mockinterview_ans": "在图像数据增强中,我通常会采用多种方法,包括但不仅限于旋转,翻转,缩放,平移和色彩抖动,以增加模型的泛化能力,转置图像通道可以通过调整图像的RGB通道顺序实现,限制像素值范围则通过归一化或标准化处理,确保所有图像数据在相同范围,学习率策略上,我会使用如余弦退火,指数衰减或嵌套周期策略,以优化模型训练过程,找到最佳的学习率,这种方式可以提高训练效率,并有助于模型收敛到最优解."}}<|action_end|>'
        subtext6_user = f'''
        根据对话上下文给出固定格式的回答，注意，一定要使用get_comments函数，函数的使用格式一定要严格符合要求
        对话上下文是：
        在你描述的"智能数据分析系统"中，数据清洗是非常关键的步骤。我想了解一下你如何利用STREAM操作进行数据清洗，特别是在去除重复数据或无关数据方面有什么经验或技巧？这将帮助我们更深入地了解你的实践能力。
        使用STREAM操作进行数据清洗时,优先评估数据流的特性,确定清洗策略,采用滑动窗口减少内存占用,实时监测数据质量,及时调整参数,去除重复数据,利用keyBy对数据进行分组,通过窗口函数比较相同键值的数据,保留最新或最常用记录,过滤无关数据,设置过滤条件,如数据类型,时间戳等,确保数据流的纯净和有效性,实践中,持续监控清洗效果,优化算法,提高数据处理效率和准确性.

        回答是：
        你回答了一道专业问题，我将使用MockInterview工具中的get_comments函数来评估你的答案，请稍等片刻{subtext6_action1}

        对话上下文是：
        从你提到的技术中，让我们先探讨关于数据增强的问题。在处理图像数据时，你如何进行数据增强，包括转置图像通道和限制像素值范围？这是理解你在深度学习中如何有效地使用数据增强技术的关键问题。期待你的详细解答。如果对于学习率策略也有相关的问题，你同样可以准备和分享，这样我们可以更全面地深入了解你在深度学习模型优化方面的知识和经验。
        在图像数据增强中,我通常会采用多种方法,包括但不仅限于旋转,翻转,缩放,平移和色彩抖动,以增加模型的泛化能力,转置图像通道可以通过调整图像的RGB通道顺序实现,限制像素值范围则通过归一化或标准化处理,确保所有图像数据在相同范围,学习率策略上,我会使用如余弦退火,指数衰减或嵌套周期策略,以优化模型训练过程,找到最佳的学习率,这种方式可以提高训练效率,并有助于模型收敛到最优解.
        
        回答是：
        你关于数据增强技术的描述非常全面，我将使用MockInterview工具中的get_comments函数来获取对你回答的深度和准确性的评价信息，请稍等{subtext6_action2}

        对话上下文是：
        {subtext4_assistant}
        {subtext5_assistant}

        回答是：
        '''
        submessages6 = [{'role': 'user', 'content': subtext6_user}]
        subtext6_assistant = deepseek_response(submessages6)

        self.messages.append({'role': 'assistant', 'content': subtext6_assistant})

        # 使用正则表达式提取 mockinterview_query 和 mockinterview_ans 的内容
        pattern = r'"mockinterview_query":\s*"([^"]*)"\s*,\s*"mockinterview_ans":\s*"([^"]*)"'
        match = re.search(pattern, subtext6_assistant)

        mockinterview_query = match.group(1)
        mockinterview_ans = match.group(2)

        print('########mockquery###########')
        print(mockinterview_query)
        print('########mockans#############')
        print(mockinterview_ans)

        url = "http://0.0.0.0:8004/rag/mockinterview_comments"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"mockinterview_query": mockinterview_query, "mockinterview_ans": mockinterview_ans})

        response = requests.post(url, data=payload, headers=headers).json()

        content = response['content']
        rightans = response['rightans']
        comments_content = response['comments']
        print('########comments_content#########')
        print(comments_content)
        
        self.messages.append({'role': 'user', 'content': f'environment name=<|plugin|>{rightans}'})
        
        subtext7_action1 = '<|action_start|><|plugin|>{"name": "MockInterview.get_comments", "parameters": {"mockinterview_query": "Redis 数据分片工具是一种用于分布式存储和查询的解决方案。请解释在实际应用中，Redis 数据分片是如何确保数据的一致性和可用性的？此外，请简述在进行 Redis 数据分片时，需要考虑的关键因素有哪些？", "mockinterview_ans": "Redis数据分片就是将数据分散存储在不同的节点上，每个节点负责一部分数据。为了确保数据的一致性和可用性，Redis使用了多种机制和技术：\n\n1. 主从复制：Redis支持主从复制模式，其中一个节点作为主节点负责写入数据，其他从节点负责从主节点同步数据。这样即使主节点出现故障，从节点也可以提供服务，保证系统的可用性。同时，从节点可以分担查询请求，提高系统的并发处理能力。\n\n2. 分片键：Redis分片是基于键的范围进行分片，每个节点负责一部分键值对。客户端在执行查询时，会根据键的范围找到对应的节点进行查询，这样可以确保每个节点的数据是一致的。\n\n3. 故障转移：当某个节点出现故障时，Redis会从其他节点中选举出一个新的主节点来代替故障节点，保证系统的可用性。\n\n在进行Redis数据分片时，需要考虑的关键因素包括：\n\n1. 分片策略：选择合适的分片策略来分散数据，避免数据热点问题。\n\n2. 数据一致性：确保分片之间的数据一致性，避免数据冲突。\n\n3. 容错性：设计容错机制，确保系统在节点故障时仍能正常工作。\n\n4. 扩展性：考虑系统的扩展性，确保在需要扩展时能够轻松地添加新的节点。\n\n5. 性能：评估分片对系统性能的影响，确保分片后的查询效率不会明显下降。"}}<|action_end|>'
        subtext7_action2 = '<|action_start|><|plugin|>{"name": "MockInterview.get_comments", "parameters": {"mockinterview_query": "在Go语言中，当一个函数从内核态返回时，它通常会回到哪个用户态的函数继续执行？请详细解释这一过程。", "mockinterview_ans": "在Go语言中，当一个函数从内核态返回时，它通常会回到调用它的用户态函数继续执行。这个过程中涉及的是系统调用和上下文切换的概念。\n\n具体来说，当一个Go程序执行系统调用时，CPU会从用户态切换到内核态。在这个状态下，操作系统的内核会处理这个系统调用，执行相应的操作。当系统调用完成后，CPU需要从内核态切换回用户态，以便程序可以继续执行。\n\n在这个切换过程中，CPU会保存当前内核态的上下文信息，包括程序计数器、寄存器状态等，以便稍后能够正确地恢复执行。然后，将对应的用户态上下文信息加载到CPU中，这样程序就可以从系统调用返回的位置继续执行。\n\n在Go语言中，这种切换过程是自动处理的，开发者不需要手动管理。Go运行时系统会负责处理系统调用和上下文切换的细节，确保程序能够正确地执行。\n\n总的来说，当一个函数从内核态返回时，它会回到调用它的用户态函数继续执行。这个过程涉及系统调用和上下文切换，是由Go运行时系统自动处理的。"}}<|action_end|>'
        subtext7_user = f'''
        对话上下文是：
        你回答了一道专业问题，我将使用MockInterview工具中的get_comments函数来评估你的答案，请稍等片刻{subtext7_action1}
        environment name=<|plugin|>Redis 数据分片工具确保数据一致性和可用性的方式：\n\n1. **复制（Replication）**：\n   - Redis 数据分片使用主从复制模式，其中一个节点（主节点）负责接收所有的写请求，并同步数据到多个从节点。这样即使主节点发生故障，从节点也能提供数据的备份，确保数据的可用性。\n\n2. **分片键（Sharding Keys）**：\n   - Redis 通过将数据分割成不同的分片，使得每个分片只存储部分数据。查询请求会被划分到对应的分片上进行处理，从而提高并发处理能力，保证数据的一致性。\n\n3. **故障转移（Failover）**：\n   - 当主节点不可用时，Redis 会自动将一个从节点升级为主节点，接管服务，从而保证系统的可用性。\n\n在进行 Redis 数据分片时，需要考虑的关键因素：\n\n1. **分片策略（Sharding Strategy）**：\n   - 选择合适的分片策略至关重要。常见的分片策略包括范围分片、哈希分片等。每种策略都有其优缺点，需根据具体业务场景选择合适的分片方式。\n\n2. **数据均衡（Data Balancing）**：\n   - 分片需要确保数据在各个分片之间均衡分布，避免出现某些分片负载过高的情况，从而保证性能和可用性。\n\n3. **扩展性（Scalability）**：\n   - 分片系统需要考虑未来的可扩展性，确保能够轻松地添加或移除节点以适应数据量的变化。\n\n4. **数据一致性（Data Consistency）**：\n   - 在分片系统中，不同分片间的数据一致性是一个挑战。需要设计合理的同步机制，确保数据在各个分片之间保持一致。\n\n5. **故障处理（Fault Tolerance）**：\n   - 分片系统需要对网络分区、节点故障等异常情况具备应对能力，确保系统的稳定性和数据的可用性。\n\n通过考虑这些关键因素，并合理配置 Redis 数据分片，可以在保证数据一致性和可用性的同时，应对大规模数据存储和查询的需求。
        
        回答是：
        从你的回答来看，你对Redis数据分片的机制以及实现数据一致性和可用性的方式有了一定的理解。你的回答中提到了主从复制、分片键和故障转移等机制，这些都是Redis数据分片中确保数据一致性和可用性的关键手段。\n\n不过，在解释数据一致性时，你提到“每个节点的数据是一致的”，这种表述可能会引起误解。实际上，在分布式系统中，由于网络延迟、节点故障等因素，确保全局数据的一致性是一个复杂的问题。Redis通过分片键的方式将数据分散在不同的节点上，每个节点只负责自己分片内的数据一致性。因此，更准确的表述应该是，通过分片键，每个节点负责的数据子集是一致的。\n\n此外，你在讨论Redis数据分片时需要考虑的关键因素时，提到了一些重要的点，比如分片策略、数据一致性、容错性和性能。这些都是确保Redis数据分片系统有效工作的关键考虑因素。然而，你遗漏了数据均衡和扩展性这两个同样重要的因素。数据均衡确保了各个分片之间负载的均衡，避免某些分片过载而其他分片闲置。扩展性则指的是系统应对数据量增长或节点故障时的能力，确保系统能够平滑扩展。\n\n总的来说，你的回答对Redis数据分片的基本概念和实现方式有了较好的把握，但在细节描述上还可以更加精准和全面。希望你在未来的学习和实践中，能够更深入地理解Redis数据分片的机制，并综合考虑各个关键因素，确保系统的稳定性和高效性。

        对话上下文是：
        你对一个专业问题进行了回答，我将使用MockInterview工具中的get_comments函数来评估你的答案，请稍等片刻{subtext7_action2}
        environment name=<|plugin|>为了回答这一问题，需要理解Go语言中的函数调用过程，包括从内核态到用户态的转换。\n\n在Go语言中，当一个函数从内核态返回时，它通常会回到用户态的函数继续执行。具体来说，这个过程涉及到以下步骤：\n\n1. **中断处理**：当程序执行到内核态时，通常是由于系统调用或中断发生。内核态的程序处理完系统调用或中断后，需要返回到用户态。\n\n2. **返回地址保存**：在从内核态返回到用户态之前，内核会将返回地址（即用户态函数的地址）保存在一个特定的地方，以便在返回时能够正确地跳转到用户态函数。\n\n3. **用户态函数恢复**：当内核完成必要的操作后，它会将控制权交还给用户态程序。此时，程序会从保存好的返回地址处开始执行，即返回到之前调用的用户态函数。\n\n4. **函数调用栈管理**：在函数调用过程中，每个函数的调用信息（包括返回地址、局部变量等）会被压入栈中。当函数返回时，这些信息会从栈中弹出，恢复到调用该函数之前的状态。\n\n5. **局部变量管理**：在用户态函数中，局部变量的生命周期由栈管理。当函数返回时，栈中的局部变量空间会被释放，以便为新的函数调用做准备。\n\n6. **系统调用返回**：如果当前函数是在系统调用后从内核态返回，那么它将直接返回到用户态的函数调用点。如果系统调用是异步的，那么返回过程可能会涉及更多的上下文切换和信号处理。\n\n综上所述，当一个函数从内核态返回时，它会返回到之前调用的用户态函数，继续执行该函数中的代码。这一过程涉及到栈的管理、返回地址的保存与恢复，以及局部变量的生命周期管理。通过这些机制，Go语言确保了函数调用的正确性和程序的正常执行。
        
        回答是：
        你的回答中提到了系统调用和上下文切换的概念，这是正确的。然而，你的回答中并没有详细解释返回地址的保存和恢复过程，以及函数调用栈的管理。这些细节对于理解函数从内核态返回后如何继续执行用户态的代码是非常重要的。\n\n正确的回答应该包括以下几个关键点：\n\n1. **中断处理**：当程序执行到内核态时，通常是由于系统调用或中断发生。内核态的程序处理完系统调用或中断后，需要返回到用户态。\n\n2. **返回地址保存**：在从内核态返回到用户态之前，内核会将返回地址（即用户态函数的地址）保存在一个特定的地方，以便在返回时能够正确地跳转到用户态函数。\n\n3. **用户态函数恢复**：当内核完成必要的操作后，它会将控制权交还给用户态程序。此时，程序会从保存好的返回地址处开始执行，即返回到之前调用的用户态函数。\n\n4. **函数调用栈管理**：在函数调用过程中，每个函数的调用信息（包括返回地址、局部变量等）会被压入栈中。当函数返回时，这些信息会从栈中弹出，恢复到调用该函数之前的状态。\n\n5. **局部变量管理**：在用户态函数中，局部变量的生命周期由栈管理。当函数返回时，栈中的局部变量空间会被释放，以便为新的函数调用做准备。\n\n6. **系统调用返回**：如果当前函数是在系统调用后从内核态返回，那么它将直接返回到用户态的函数调用点。如果系统调用是异步的，那么返回过程可能会涉及更多的上下文切换和信号处理。\n\n通过这些机制，Go语言确保了函数调用的正确性和程序的正常执行。\n\n因此，虽然你的回答提到了一些相关概念，但缺乏对关键细节的详细解释。建议在回答问题时，提供更加全面的解释，以确保面试者能够充分理解这一过程。

        对话上下文是：
        {subtext6_assistant}
        environment name=<|plugin|>{content}

        回答是：
        {comments_content}
        '''

        # submessages7 = [{'role': 'user', 'content': subtext7_user}]
        # subtext7_assistant = deepseek_response(submessages7)

        self.messages.append({'role': 'assistant', 'content': comments_content})

        continue_content = deepseek_response([{'role': 'user', 'content': '改写这句话：好的好的，感谢您的点评，请继续面试。\n要求不改变原句意思，语言简洁，只输出你改写后的句子，不要其他无关内容'}])
        self.messages.append({'role': 'user', 'content': continue_content})

        res = deepseek_response(self.messages)
        self.messages.append({'role': 'assistant', 'content': res})
         
        return self.messages

    def save_mockinterview_messages(self):
        for i in range(500):
            try:
                messages = self.get_mockinterview_messages()
            except:
                continue
            self.convert_to_finetuneform(messages)
            self.save_json('storage/multi_chat_mi.json')

    def get_quicklyqa_messages(self, query, ans, content, comments):
        self.init_messages()
        self.messages.append({'role': 'user', 'content': '你好！'})
        self.messages.append({'role': 'assistant', 'content': '你好！有什么我可以帮助你的吗？你可以直接对我提问。我也可以作为你的复习伙伴，为你提供模拟面试或者快问快答服务。'})
        self.messages.append({'role': 'user', 'content': '开始快问快答'})
        self.messages.append({'role': 'assistant', 'content': '''好的，接下来将开始快问快答，首先，我将使用QuicklyQA工具中的get_query函数来获取一道问题，请稍等片刻<|action_start|><|plugin|>
{"name": "QuicklyQA.get_query", "parameters": {}}<|action_end|>'''})

        self.messages.append({'role': 'user', 'content': f'environment name=<|plugin|>{query}'})
        self.messages.append({'role': 'assistant', 'content': f'请问，{query}'})
        self.messages.append({'role': 'user', 'content': f'{ans}'})
        action_content = '''<|action_start|><|plugin|>{"name": "QuicklyQA.get_comments", "parameters": {"quicklyQA_query": "%s", "quicklyQA_ans": "%s"}}<|action_end|>''' % (query, ans)
        self.messages.append({'role': 'assistant', 'content': f'我将使用QuicklyQA工具中的get_comments函数来获取上下文内容评估你的回答，请稍等片刻{action_content}'})
        self.messages.append({'role': 'user', 'content': f'environment name=<|plugin|>{content}'})
        self.messages.append({'role': 'assistant', 'content': f'{comments}'})
        self.messages.append({'role': 'user', 'content': '了解了，下一题'})
        self.messages.append({'role': 'assistant', 'content': '''<|action_start|><|plugin|>{"name": "QuicklyQA.get_query", "parameters": {}}<|action_end|>'''})
        return self.messages

    def save_quicklyqa_messages(self):
        with open(file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)

        for i in range(0, len(datas), 2):
            # 解析 JSON 数据
            conversation1 = datas[i]['conversation'][0]
            conversation2 = datas[i+1]['conversation'][0]
            # 提取问题、用户回答和点评
            query = conversation1['input'].strip()
            content_text = conversation1['output'].strip()
    
            ans = conversation2['input'].strip()
            comments = conversation2['output'].strip()
            messages = self.get_quicklyqa_messages(query, ans, content_text, comments)
            self.convert_to_finetuneform(messages)
        
            self.save_json('storage/multi_chat_qa.json')

if __name__ == "__main__":
    qacomment = QAComment()
    qacomment.get_questiondb()
    qacomment.get_contentdb()
    # 调用多线程处理并分批保存
    # qacomment.process_queries_in_parallel(MESSAGE2_PROMPT, MESSAGE3_PROMPT, file_path)
    # getdata = Getdatas()

    # getdata.save_quicklyqa_messages()
    # getdata.save_mockinterview_messages()

