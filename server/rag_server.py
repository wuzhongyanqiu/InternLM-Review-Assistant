from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

from rag_worker import GetContent, GoogleSearch
from minerU_worker import MinerU
from simpleparse_worker import SimpleParse
from concurrent.futures import ThreadPoolExecutor

from lmdeploy.serve.openai.api_client import APIClient
import os
import sqlite3
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

api_key = ''

api_key = os.getenv('GOOGLE_SEARCH_KEY', '') 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deepseek_response(messages):
    from openai import OpenAI

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
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

def get_response(messages):
    from openai import OpenAI

    client = OpenAI(api_key="sk-abfghsaljbqtabrkjeexxwegooutwyovhsgywhzrjbxdotws", base_url="https://api.siliconflow.cn/v1")

    response = client.chat.completions.create(
        model='internlm/internlm2_5-7b-chat',
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content

app = FastAPI()
getcontent = GetContent()
minerU = MinerU()
simpleparse = SimpleParse()

current_dir = os.path.dirname(os.path.abspath(__file__))

DIR_PATH = os.path.join(current_dir, "../tmp_dir/datas")
MINERU_OUTPUTDIR = os.path.join(current_dir, "MinerU_output")
RESUME_PATH = os.path.join(current_dir, "../tmp_dir/resume/resume.pdf")
DB_PATH = os.path.join(current_dir, "storage/database")

import fitz  

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    
    extracted_text = ""
    
    for page_num in range(document.page_count):
        page = document.load_page(page_num)

        text = page.get_text()
        extracted_text += text
    
    document.close()

    return extracted_text

class RagItem(BaseModel):
    mockinterview_query: str = Field(default='')
    mockinterview_ans: str = Field(default='')
    mockinterview_keywords: str = Field(default='')
    quicklyQA_query: str = Field(default='')
    quicklyQA_ans: str = Field(default='')

def clear_database(db_filename=DB_PATH):
    if os.path.exists(db_filename):
        os.remove(db_filename)
        print(f"Database {db_filename} has been deleted.")
    else:
        print(f"Database {db_filename} does not exist.")

def store_questions_in_db(questions, db_filename=DB_PATH):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL
    )
    ''')

    for question in questions:
        cursor.execute('''
        INSERT INTO questions (question) VALUES (?)
        ''', (question,))
    
    conn.commit()
    conn.close()

def renumber_questions(cursor):
    # 获取所有的问题按 id 顺序
    cursor.execute('SELECT id FROM questions ORDER BY id')
    rows = cursor.fetchall()

    # 暂时禁用外键约束（如果有外键）
    cursor.execute('PRAGMA foreign_keys = OFF')

    # 更新每条记录的 id，按行的顺序重新编号
    new_id = 1
    for row in rows:
        old_id = row[0]
        cursor.execute('UPDATE questions SET id = ? WHERE id = ?', (new_id, old_id))
        new_id += 1

    # 重新启用外键约束
    cursor.execute('PRAGMA foreign_keys = ON')

def check_question(question_id, question_text):
    system_prompt = '判断给出的句子是否是一个表达清楚、明确的专业性面试题，你的标准非常严格，如果不是，输出 None'
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text}
    ]
    response = deepseek_response(messages)
    return question_id, response

def check_questions(db_filename=DB_PATH):
    db_batch_size = 100
    
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # 查询所有的问题
    cursor.execute('SELECT * FROM questions')
    rows = cursor.fetchall()

    count = 0  # 用于计数已处理的问题数量
    # 使用线程池进行并发处理
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_question, row[0], row[1]): row for row in rows}
        for future in futures:
            question_id, response = future.result()
            if 'None' in response:  # 你的判断函数
                cursor.execute('DELETE FROM questions WHERE id = ?', (question_id,))
                print(f'Deleted ID: {question_id}, Question: {futures[future][1]}')
            
            count += 1
            if count % db_batch_size == 0:  # 每处理 batch_size 个问题提交一次
                conn.commit()
                print(f"已处理 {count} 个问题，并提交更改")

    # 提交剩余的更改
    conn.commit()

    # 删除后重新编号
    renumber_questions(cursor)

    # 提交编号更改并关闭
    conn.commit()
    conn.close()

def process_data(data):
    content = data.page_content
    messages = [
        {"role": "system", "content": "你擅长根据段落信息总结出若干道面试题，注意，仅给出用逗号分隔的面试题，不要说其他无关的语句，如果给定的信息不适合总结面试题，输出 None"},
        {"role": "user", "content": "段落信息为：\n{content}\n".format(content=content)}
    ]

    logger.info(f"Sending messages to API: {messages}")

    response = deepseek_response(messages)

    logger.info(f"API response: {response}")

    keywords_string = response

    keywords_list = [keyword.strip() for keyword in keywords_string.split(',')]
    keywords_list = [keyword for sublist in keywords_list for keyword in sublist.split('\n')]
    keywords_list = [keyword.strip() for keyword in keywords_list if keyword.strip()]

    questions = []
    for keyword in keywords_list:
        if keyword and len(keyword) > 8:  # Adjust the length as needed
            questions.append(keyword)
    
    store_questions_in_db(questions)  # 问题存储到数据库

def process_all_datas_concurrently(datas):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_data = {executor.submit(process_data, data): data for data in datas}
        
        for future in concurrent.futures.as_completed(future_to_data):
            data = future_to_data[future]
            try:
                data_questions = future.result()
            except Exception as exc:
                logger.error(f"Data {data} generated an exception: {exc}")

@app.post("/rag/gen_questiondb")
def generate_questiondatabase():
    try:
        # datas = simpleparse.process_folder(folder_path=DIR_PATH)
        
        # logger.info(f"Parsed data: {datas}")
        # print(f"Parsed data: {datas}")

        # print(f"Parsed data size: {len(datas)}")

        # 并发处理生成问题并存储到数据库
        # process_all_datas_concurrently(datas)

        # check_questions(db_filename=DB_PATH)

        return {"message": "questiondatabase generated successfully."}
    except Exception as e:
        logger.error(f"Error generating questiondatabase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/gen_contentdb")
def generate_contentdatabase():
    try:
        datas = simpleparse.process_folder(folder_path=DIR_PATH)
        
        logger.info(f"Parsed data: {datas}")
        print(f"Parsed data: {datas}")

        print(f"Parsed data size: {len(datas)}")

        getcontent.gen_contentdb(datas)
        print(f'getcontent.gen_contentdb is finish')

        # 生成问题数据库
        getcontent.gen_questiondb()

        return {"message": "contentdatabase generated successfully."}
    except Exception as e:
        logger.error(f"Error generating contentdatabase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("rag/clear_db")
def clear_alldatabase():
    try:
        clear_database()
        getcontent.del_db()
        return {"message": "delete database is successfully."}
    except Exception as e:
        logger.error(f"Error clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/mockinterview_comments")
def get_comments(ragitem: RagItem):
    logger.info(f"Received request for comments with ragitem: {ragitem}")
    print(f"Received request for comments with ragitem: {ragitem}")
    print(f"RAGITEM.ans: {ragitem.mockinterview_ans}")
    print(f"RAGITEM.query: {ragitem.mockinterview_query}")
    content = getcontent.reply_comments(ragitem.mockinterview_query+"\n"+ragitem.mockinterview_ans)
    # 搜索引擎增强生成
    if api_key:
        search_engine = GoogleSearch(api_key)
        status_code, response = search_engine.search(f"{ragitem.mockinterview_query}")
        if status_code == 200:
            parsed_results = search_engine.parse_results(response)
            for snippet in parsed_results:
                content += f'\n{snippet}'
        else:
            print(f'Errot: {response}')

    print(f"CONTNET: {content}")
    messages = [
        {"role": "system", "content": "根据你已有的知识和上下文内容，回答问题，要求语言简洁通顺，答案准确无误，注意，当上下文内容与问题无关时，不要编造答案"},
        {"role": "user", "content": f"问题：\n{ragitem.mockinterview_query}\n上下文内容：\n{content}\n"}
    ]

    logger.info(f"Sending messages to API: {messages}")
    print(f"Sending messages to API: {messages}")

    response = deepseek_response(messages)

    logger.info(f"API response: {response}")

    rightans = response

    logger.info(f"Right answer: {rightans}")
    print(f"Right answer: {rightans}")

    _messages = [
        {"role": "system", "content": "根据你已有的知识和参考答案，对面试者的答案进行点评，你直接与面试者对话，因此要符合对话的语法，如果面试者回答过于简单或者错误百出，你会严格的进行批评和指正"},
        {"role": "user", "content": f"问题:\n{ragitem.mockinterview_query}\n正确答案：\n{rightans}\n面试者的答案: \n{ragitem.mockinterview_ans}\n"}
    ] 

    logger.info(f"Sending messages to API for comments: {_messages}")
    print(f"Sending messages to API for comments: {_messages}")

    response = deepseek_response(_messages)

    logger.info(f"API response: {response}")
    print(f"API response: {response}")

    comments = response
    logger.info(f"Generated comments: {comments}")
    print(f"Generated comments: {comments}")

    return {'content': content, 'rightans': rightans, 'comments': comments}

@app.post("/rag/mockinterview_questions")
def get_questions(ragitem: RagItem):
    logger.info(f"Received request for questions with ragitem: {ragitem}")
    print(f"Received request for questions with ragitem: {ragitem}")
    keywords_str = ragitem.mockinterview_keywords

    print(f"RAGITEM.KEYWORDS: {ragitem.mockinterview_keywords}")

    keywords_list = keywords_str.split(',')

    print(f"KEYWORD_LIST: {keywords_list}")

    interview_questions = {}
    for keyword in keywords_list:
        print(f"KEYWORD: {keyword}")
        questions = getcontent.reply_questions(keyword)
        print(f"RAG_QUESTIONS: {questions}")

        interview_questions[keyword] = questions


    questions = "\n".join([f"关于{keyword}的面试题包括: \n{questions}" for keyword, questions in interview_questions.items()])
    logger.info(f"Generated interview questions: {questions}")
    print(f"Generated interview questions: {questions}")
    
    return {'question': questions}

def summarize_resume(resume_content):
    PROMPT_TEMPLATE = """
    请总结以下简历内容的专业部分，包括项目经历、科研经历、实习经历、技术栈。忽略基本信息（如姓名、邮箱、电话、教育经历）和泛泛的描述。请确保输出格式清晰。

    格式：
    项目经历: 
    科研经历: 
    实习经历:
    技术栈：

    简历内容：
    {resume_content}
    """
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(resume_content=resume_content)}
    ]

    response = deepseek_response(messages)

    return response

@app.post("/rag/mockinterview_resumes")
def get_resumes():
    # 示例提示词，关注专业内容
    resumes_content = ''
    resumes = extract_text_from_pdf(RESUME_PATH)
    resumes_content = summarize_resume(resumes)
    return {'resumes_content': resumes_content}

@app.post("/rag/quicklyQA_comments")
def get_comments(ragitem: RagItem):
    logger.info(f"Received request for comments with ragitem: {ragitem}")
    print(f"Received request for comments with ragitem: {ragitem}")
    print(f"RAGITEM.ans: {ragitem.quicklyQA_ans}")
    print(f"RAGITEM.query: {ragitem.quicklyQA_query}")
    content = getcontent.reply_comments(ragitem.quicklyQA_query+"\n"+ragitem.quicklyQA_ans)
    # 搜索引擎增强生成
    if api_key:
        search_engine = GoogleSearch(api_key)
        status_code, response = search_engine.search(f"{ragitem.mockinterview_query}")
        if status_code == 200:
            parsed_results = search_engine.parse_results(response)
            for snippet in parsed_results:
                content += f'\n{snippet}'
        else:
            print(f'Errot: {response}')
    print(f"CONTNET: {content}")
    messages = [
        {"role": "system", "content": "根据你已有的知识和上下文内容，回答问题，要求语言简洁通顺，答案准确无误，注意，当上下文内容与问题无关时，不要编造答案"},
        {"role": "user", "content": f"问题：\n{ragitem.quicklyQA_query}\n上下文内容：\n{content}\n"}
    ]

    logger.info(f"Sending messages to API: {messages}")
    print(f"Sending messages to API: {messages}")

    response = deepseek_response(messages)

    logger.info(f"API response: {response}")

    rightans = response

    logger.info(f"Right answer: {rightans}")
    print(f"Right answer: {rightans}")

    _messages = [
        {"role": "system", "content": "根据你已有的知识和参考答案，对面试者的答案进行点评，你直接与面试者对话，因此要符合对话的语法，如果面试者回答过于简单或者错误百出，你会严格的进行批评和指正"},
        {"role": "user", "content": f"问题:\n{ragitem.quicklyQA_query}\n正确答案：\n{rightans}\n面试者的答案: \n{ragitem.quicklyQA_ans}\n"}
    ] 

    logger.info(f"Sending messages to API for comments: {_messages}")
    print(f"Sending messages to API for comments: {_messages}")

    response = deepseek_response(_messages)

    logger.info(f"API response: {response}")
    print(f"API response: {response}")

    comments = response
    logger.info(f"Generated comments: {comments}")
    print(f"Generated comments: {comments}")

    return {'content': content, 'rightans': rightans, 'comments': comments}

@app.post("/rag/quicklyQA_questions")
def get_questions():
    try:
        question = getcontent.select_questions()  
        return {'query': question}
    except Exception as e:
        logger.error(f"Error get_questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # import uvicorn
    # uvicorn(app, host="0.0.0.0", port=8004)
    resumes_content = get_resumes()
    print(resumes_content)