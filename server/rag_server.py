from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

from rag_worker import GetContent
from minerU_worker import MinerU

from lmdeploy.serve.openai.api_client import APIClient
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_client = APIClient('http://127.0.0.1:23333')
model_name = api_client.available_models[0]

app = FastAPI()
getcontent = GetContent()
minerU = MinerU()

current_dir = os.path.dirname(os.path.abspath(__file__))

DIR_PATH = os.path.join(current_dir, "../tmp_dir")
MINERU_OUTPUTDIR = os.path.join(current_dir, "MinerU_output")
RESUME_PATH = os.path.join(current_dir, "../tmp_dir/resume/resume.pdf")

import fitz  

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    
    extracted_text = ""
    
    for page_num in range(document.page_count):
        page = document.load_page(page_num)

        text = page.get_text()
        extracted_text += text
    
    document.close()

    return extract_text

class RagItem(BaseModel):
    query: str = Field(default='')
    ans: str = Field(default='')
    chat_content: str = Field(default='')

@app.post("/rag/gendb")
async def generate_database():
    try:
        datas = minerU.parse_dir(dir_path=DIR_PATH, output_dir=MINERU_OUTPUTDIR)
        logger.info(f"Parsed data: {datas}")
        print(f"Parsed data: {datas}")
        getcontent.gen_contentdb(datas)
        questions = []
        for data in datas:
            messages = [
                {"role": "system", "content": "你擅长根据段落信息总结出若干道面试题，用逗号分隔"},
                {"role": "user", "content": "段落信息为：\n{content}\n".format(content=data)}
            ]

            logger.info(f"Sending messages to API: {messages}")
            print(f"Sending messages to API: {messages}")

            for item in api_client.chat_completions_v1(model=model_name, messages=messages):
                pass

            response = item
            
            logger.info(f"API response: {response}")
            print(f"API response: {response}")

            keywords_string = item['choices'][0]['message']['content']

            keywords_list = [keyword.strip() for keyword in keywords_string.split(',')]
            keywords_list = [keyword for sublist in keywords_list for keyword in sublist.split('\n')]
            keywords_list = [keyword.strip() for keyword in keywords_list if keyword.strip()]

            questions.extend(keywords_list)

        getcontent.gen_questiondb(questions)
        logger.info(f"Generated questions: {questions}")
        print(f"Generated questions: {questions}")
        return {"message": "RAG_Database generated successfully."}
    except Exception as e:
        logger.error(f"Error generating database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/comments")
async def get_comments(ragitem: RagItem):
    logger.info(f"Received request for comments with ragitem: {ragitem}")
    print(f"Received request for comments with ragitem: {ragitem}")
    content = getcontent.reply_comments(ragitem.query)
    messages = [
        {"role": "system", "content": "根据你已有的知识和上下文内容，回答问题，要求语言简洁通顺，答案准确无误，注意，当上下文内容与问题无关时，不要编造答案"},
        {"role": "user", "content": "问题：\n{question}\n上下文内容：\n{content}\n".format(question=ragitem.query, content=content)}
    ]

    logger.info(f"Sending messages to API: {messages}")
    print(f"Sending messages to API: {messages}")

    for item in api_client.chat_completions_v1(model=model_name, messages=messages):
                pass

    response = item

    logger.info(f"API response: {response}")
    print(f"API response: {response}")

    rightans = item['choices'][0]['message']['content']

    logger.info(f"Right answer: {rightans}")
    print(f"Right answer: {rightans}")

    _messages = [
        {"role": "system", "content": "根据你已有的知识和正确答案，对面试者的答案进行点评"},
        {"role": "user", "content": "正确答案：\n{rightans}\n面试者的答案: \n{ans}\n".format(rightans=rightans, ans=ragitem.ans)}
    ] 

    logger.info(f"Sending messages to API for comments: {_messages}")
    print(f"Sending messages to API for comments: {_messages}")

    for item in api_client.chat_completions_v1(model=model_name, messages=messages):
        pass

    response = item

    logger.info(f"API response: {response}")
    print(f"API response: {response}")

    comments = item['choices'][0]['message']['content']
    logger.info(f"Generated comments: {comments}")
    print(f"Generated comments: {comments}")

    return {'comments': comments}

@app.post("/rag/questions")
async def get_questions(ragitem: RagItem):
    logger.info(f"Received request for questions with ragitem: {ragitem}")
    print(f"Received request for questions with ragitem: {ragitem}")
    messages = [
        {"role": "system", "content": "你擅长提取段落的专业名词，以短句的方式输出，用逗号分隔。注意：你只提取专业名词，普遍的名词不提取"},
        {"role": "user", "content": "你要切分成短句的段落为：\n{content}\n".format(content=ragitem.chat_content)}
    ]

    logger.info(f"Sending messages to API for keywords: {messages}")
    print(f"Sending messages to API for keywords: {messages}")

    for item in api_client.chat_completions_v1(model=model_name, messages=messages):
        pass

    response = item
    
    logger.info(f"API response: {response}")
    print(f"API response: {response}")
 
    keywords_string = item['choices'][0]['message']['content']
    keywords_list = [keyword.strip() for keyword in keywords_string.split(',')]

    interview_questions = {}
    for keyword in keywords_list:
        questions = getcontent.reply_questions(keyword)
        print("rag_questions=")
        print(questions)
        interview_questions[keyword] = questions

    questions = "\n".join([f"关于{keyword}的面试题包括: \n{questions}" for keyword, questions in interview_questions.items()])
    logger.info(f"Generated interview questions: {questions}")
    print(f"Generated interview questions: {questions}")

    messages = [
        {"role": "system", "content": "你能根据上下文语境，从给定的面试题中选择一道最适合的来询问面试者，注意，你只问一道面试题，而且你需要结合上下文语境进行提问。"},
        {"role": "user", "content": "上下文内容是:\n{content}\n，给定的面试题是：\n{questions}\n".format(content=ragitem.chat_content, questions=questions)}
    ]

    logger.info(f"Sending messages to API to select question: {messages}")
    print(f"Sending messages to API to select question: {messages}")

    for item in api_client.chat_completions_v1(model=model_name, messages=messages):
        pass

    response = item
    
    logger.info(f"API response: {response}")
    print(f"API response: {response}")

    question = item['choices'][0]['message']['content']
    logger.info(f"Selected question: {question}")
    print(f"Selected question: {question}")
    
    return {'question': question}

@app.post("/rag/resumes")
async def get_resumes():
    try:
        resumes_content = extract_text_from_pdf(RESUME_PATH)
    except:
        return {'Error': '未上传简历'}
    return {'resumes_content': resumes_content}


if __name__ == "__main__":
    import uvicorn
    uvicorn(app, host="0.0.0.0", port=8004)