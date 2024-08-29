import sys
sys.path.append("/root/Mock-Interviewer")
import sqlite3
from rag.rerank_model import reRankLLM
from rag.faiss_retriever import FaissRetriever
from rag.bm25_retriever import BM25
from server.tools.tools_prompt import *
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from rag.parse_knowledge import process_folder
import os
from typing import Union, List

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 问题 database 路径
DB_PATH = os.path.join(current_dir, "../../storage/db_questions.db")

# embedding model路径
EMBED_MODEL1_PATH = "moka-ai/m3e-base"
EMBED_MODEL2_PATH = "thenlper/gte-large"
EMBED_MODEL3_PATH = "BAAI/bge-large-zh-v1.5"
EMBED_MODEL4_PATH = "/root/models/bce-embedding-base_v1"

# rerank model路径
RERANK_MODEL1_PATH = "BAAI/bge-reranker-large"
RERANK_MODEL2_PATH = "/root/models/bce-reranker-base_v1"

# 处理文件路径
DATAS_FOLDER_PATH = os.path.join(current_dir, "../../datas/")

# 持久化路径
PERSIST_PATH = os.path.join(current_dir, "../../storage/")

def load_data(file_path):
    # 初始化一个空列表来存储文件中的每一行
    data = []

    # 使用 'with' 语句打开文件，确保文件会在操作完成后正确关闭
    with open(file_path, "r", encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 去除行尾的换行符，并添加到列表中
            data.append(line.strip("\n"))
    return data

def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for faiss in faiss_ans:
        for doc, score in faiss:
            items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]

    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

# embedding召回方法
def embedding_retriever(faissretriever, query):
    max_length = 2000
    faiss_context = faissretriever.GetTopK(query, 5)
    faiss_min_score = 0.0
    if(len(faiss_context) > 0):
        faiss_min_score = faiss_context[0][1]
    cnt = 0
    emb_ans = ""
    for doc, score in faiss_context:
        cnt = cnt + 1
        # 最长选择max length
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
        # 最多选择6个
        if(cnt > 2):
            break
    return faiss_context, emb_ans

# 文本召回方法
def text_retriever(retriever, query):
    max_length = 2000
    retriever_context = retriever.GetBM25TopK(query, 5)
    retriever_ans = ""
    cnt = 0
    for doc in retriever_context:
        cnt = cnt + 1
        if(len(retriever_ans + doc.page_content) > max_length):
            break
        retriever_ans = retriever_ans + doc.page_content
        if(cnt > 2):
            break
    return retriever_context, retriever_ans

class SelectQuestionTool():
    def __init__(self):
        self.description = "用于从数据库查找问题"
        self.result_prompt = transquestion_prompt_template

    def reply(self) -> str:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        response_str = result[1].strip()
        ans = transquestion_prompt_template.format(response_str)

        return ans

class AnswerEvaluationTool():
    def __init__(self):
        self.description = "用于调用RAG得到相关的上下文"
        self.result_prompt = interview_prompt_input

        self.embed_model1_path = EMBED_MODEL1_PATH
        self.embed_model2_path = EMBED_MODEL2_PATH
        self.embed_model3_path = EMBED_MODEL3_PATH
        self.embed_model4_path = EMBED_MODEL4_PATH
        self.data = process_folder(DATAS_FOLDER_PATH)
        self.rerank_path1 = RERANK_MODEL1_PATH
        self.rerank_path2 = RERANK_MODEL2_PATH
        self.persist_path1 = PERSIST_PATH + self.embed_model1_path
        self.persist_path2 = PERSIST_PATH + self.embed_model2_path 
        self.persist_path3 = PERSIST_PATH + self.embed_model3_path 
        self.persist_path4 = PERSIST_PATH + self.embed_model4_path 
        self.faiss_retriever1 = FaissRetriever(self.embed_model1_path, self.data, self.persist_path1)
        self.faiss_retriever2 = FaissRetriever(self.embed_model2_path, self.data, self.persist_path2)
        self.faiss_retriever3 = FaissRetriever(self.embed_model3_path, self.data, self.persist_path3)
        self.faiss_retriever4 = FaissRetriever(self.embed_model4_path, self.data, self.persist_path4)
        self.bm25 = BM25(self.data)
        self.rerank1 = reRankLLM(self.rerank_path1)
        self.rerank2 = reRankLLM(self.rerank_path2)

    def reply(self, query, ans, rag_content):
        faiss_context1 = self.faiss_retriever1.GetTopK(rag_content, 2)
        faiss_context2 = self.faiss_retriever2.GetTopK(rag_content, 2)
        faiss_context3 = self.faiss_retriever3.GetTopK(rag_content, 2)
        faiss_context4 = self.faiss_retriever4.GetTopK(rag_content, 2)
        faiss_content = [faiss_context1, faiss_context2, faiss_context3, faiss_context4]
        bm25_context = self.bm25.GetBM25TopK(rag_content, 2)      
        rerank_ans1 = reRank(self.rerank1, 2, rag_content, bm25_context, faiss_content)
        rerank_ans2 = reRank(self.rerank2, 2, rag_content, bm25_context, faiss_content)
        ans = self.result_prompt.format(query, ans, rerank_ans1)
        return ans

    def test_rag(self, query):
        # faiss召回topk
        faiss_context1, emb_ans1 = embedding_retriever(self.faiss_retriever1, query) 
        faiss_context2, emb_ans2 = embedding_retriever(self.faiss_retriever2, query) 
        faiss_context3, emb_ans3 = embedding_retriever(self.faiss_retriever3, query) 
        faiss_context4, emb_ans4 = embedding_retriever(self.faiss_retriever4, query) 

        # bm2.5召回topk
        bm25_context, bm25_ans = text_retriever(self.bm25, query)

        faiss_content = [faiss_context1, faiss_context2, faiss_context3, faiss_context4]    
        rerank_ans1 = reRank(self.rerank1, 2, query, bm25_context, faiss_content)
        rerank_ans2 = reRank(self.rerank2, 2, query, bm25_context, faiss_content)
        return rerank_ans1, rerank_ans2, emb_ans1, emb_ans2, emb_ans3, emb_ans4, bm25_ans

class ParsingResumesTool():
    def __init__(self):
        self.description = "用于解析简历PDF"

    def reply(self, document_path):     
        try:
            text = extract_text(document_path)
            return text
        except Exception as e:
            return f"解析PDF时发生错误: {e}"

if __name__ == "__main__":
    answerevaluationtool = AnswerEvaluationTool()