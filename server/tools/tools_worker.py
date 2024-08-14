import sys
sys.path.append("/root/Mock-Interviewer/")
import sqlite3
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from rag.rerank_model import reRankLLM
from rag.faiss_retriever import FaissRetriever
from rag.bm25_retriever import BM25
from server.tools.tools_prompt import *
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from rag.parse_knowledge import process_folder
from server import ServerConfigs

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
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

class SelectQuestionTool():
    def __init__(self):
        self.description = "用于从数据库查找问题"
        self.result_prompt = transquestion_prompt_template

    def reply(self) -> str:
        conn = sqlite3.connect('file:/root/Mock-Interviewer/datas_processed/db_questions.db?mode=ro', uri=True)
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
        self.result_prompt = interview_prompt_template

        self.embed_model_path = ServerConfigs.EMBED_MODEL_PATH
        self.data = process_folder(ServerConfigs.DATAS_FOLDER_PATH)
        self.rerank_path = ServerConfigs.RERANK_MODEL_PATH
        self.persist_path = ServerConfigs.PERSIST_PATH + self.embed_model_path 
        self.faiss_retriever = FaissRetriever(self.embed_model_path, self.data, self.persist_path)
        self.bm25 = BM25(self.data)
        self.rerank = reRankLLM(self.rerank_path)

    def reply(self, query, ans):
        faiss_context = self.faiss_retriever.GetTopK(query, 3)
        bm25_context = self.bm25.GetBM25TopK(query, 3)      
        rerank_ans = reRank(self.rerank, 5, query, bm25_context, faiss_context)
        ans = self.result_prompt.format(query, ans, rerank_ans)
        return ans

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
    parsingresumestool = ParsingResumesTool()
    document_path = "./resume.pdf"
    result = parsingresumestool.reply(document_path)
    print(result)